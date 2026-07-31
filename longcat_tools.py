"""LongCat-Next tool-calling: build the canonical tool prompt (TypeScript-namespace format the
model was trained on) and parse its <longcat_tool_call> XML output into OpenAI tool_calls.
Ported faithfully from the LongCat-Next inference recipe (longcat_prompt_builder.py /
longcat_xml_detector.py / the model's own parse_model_response.py)."""
import re, json, uuid

# ---- prompt building: OpenAI function schema -> TypeScript namespace (functions2typescript) ----
def _param_convert(param_name, param_info, required_params, indent_str=" " * 8, is_return_type=False):
    optional = "" if param_name in required_params else "?"
    param_type = param_info.get("type", "string")
    if param_type == "integer":
        ts_type = "number"
    elif param_type == "object":
        ts_params = []
        for pn, pi in param_info.get("properties", {}).items():
            ts_params.append(_param_convert(pn, pi, param_info.get("required", []), indent_str + " " * 4))
        ts_type = "{\n" + ",\n".join(ts_params) + "\n" + indent_str + "}"
    elif "enum" in param_info:
        ts_type = '"' + '" | "'.join(param_info["enum"]) + '"'
    elif param_type == "array":
        if "items" in param_info:
            item_type = param_info["items"].get("type", "any")
            if item_type == "object":
                ts_params = []
                for pn, pi in param_info["items"].get("properties", {}).items():
                    ts_params.append(_param_convert(pn, pi, param_info["items"].get("required", []), indent_str + " " * 4))
                item_type = "{\n" + ",\n".join(ts_params) + "\n" + indent_str + "}"
            ts_type = item_type + "[]"
        else:
            ts_type = param_type
    else:
        ts_type = param_type
    ts_desc = param_info.get("description", "").replace("\n", " ")
    if "example_value" in param_info:
        ts_desc = "%s, example_value: %s" % (ts_desc, param_info["example_value"])
    if is_return_type:
        return ("%s; // %s" % (ts_type, ts_desc)) if ts_desc else ("%s;" % ts_type)
    if ts_desc:
        return "%s// %s\n%s%s%s: %s" % (indent_str, ts_desc, indent_str, param_name, optional, ts_type)
    return "%s%s%s: %s" % (indent_str, param_name, optional, ts_type)


def _functions2typescript(functions):
    if not isinstance(functions, list):
        functions = [functions]
    out = []
    for f in functions:
        params = f.get("parameters", {}) or {}
        req = params.get("required", [])
        ts_params = ",\n".join(_param_convert(pn, pi, req) for pn, pi in params.get("properties", {}).items())
        out.append("\n    // %s\n    type %s = (_:{\n%s\n    }) => any;" % (f.get("description", ""), f["name"], ts_params))
    return "\n".join(out)


_MULTI_TOOL = """
    ## multi_tool_use

    namespace multi_tool_use {
        // Run multiple functions tools in parallel when they can operate independently.
        type parallel = (_: {
            tool_uses: { recipient_name: string, parameters: object }[],
        }) => any;
    } // namespace multi_tool_use
"""


def build_tools_system_block(tools):
    """The canonical '# Tools' system block (TS-namespace functions + multi_tool_use)."""
    block = "# Tools\n"
    has_fn = False
    for t in tools or []:
        fn = t.get("function") if t.get("type") == "function" else (t if "name" in t else None)
        if fn:
            block += "\n    ## functions\n\n    namespace functions {\n%s\n\n    }// namespace functions\n" % _functions2typescript(fn)
            has_fn = True
    if has_fn:
        block += _MULTI_TOOL
    return block


# ---- output parsing: <longcat_tool_call> -> OpenAI tool_calls ----
# The model emits TWO syntaxes inside <longcat_tool_call> (prompt-dependent, both trained):
#   1. XML args:  name\n<longcat_arg_key>k</longcat_arg_key><longcat_arg_value>v</longcat_arg_value>...
#   2. TS call :  functions.name({key: "value", ...})   — JS object literal (keys often
#      unquoted), and generation may STOP right after ')' with no closing tag.
# So the block regex tolerates a missing </longcat_tool_call>, and each block tries the
# XML pair syntax first, then the TS-call syntax.
_TC = re.compile(r"<longcat_tool_call>(.*?)(?:</longcat_tool_call>|(?=<longcat_tool_call>)|$)", re.DOTALL)
_PAIR = re.compile(r"<longcat_arg_key>(.*?)</longcat_arg_key>\s*<longcat_arg_value>(.*?)</longcat_arg_value>", re.DOTALL)
_TS_CALL = re.compile(r"^\s*([\w.]+)\s*\((.*)\)\s*$", re.DOTALL)
_UNQUOTED_KEY = re.compile(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:')


def _parse_object_literal(s):
    """JS-ish object literal -> dict (strict JSON first, then quote bare keys). None on failure."""
    s = s.strip()
    if not s:
        return {}
    for candidate in (s, _UNQUOTED_KEY.sub(r'\1"\2":', s)):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    return None


def _arg_type(name, key, tools):
    for t in tools or []:
        fn = t.get("function") if t.get("type") == "function" else t
        if fn and fn.get("name") == name:
            return fn.get("parameters", {}).get("properties", {}).get(key, {}).get("type")
    return None


def _strip_ns(name):
    # model calls via the TS namespace ("functions.get_weather"); OpenAI clients expect the
    # bare declared name -> strip a leading "functions." / "multi_tool_use." namespace prefix.
    for ns in ("functions.", "multi_tool_use."):
        if name.startswith(ns):
            return name[len(ns):]
    return name


def _one_call(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def parse_tool_calls(text, tools):
    """Return (normal_text, tool_calls[]). tool_calls in OpenAI shape (arguments = JSON string)."""
    if "<longcat_tool_call>" not in text:
        return text, []
    idx = text.find("<longcat_tool_call>")
    normal = text[:idx].strip()
    calls = []
    for block in _TC.findall(text):
        block = block.strip()
        pairs = _PAIR.findall(block)
        if pairs:  # syntax 1: XML arg pairs
            m = re.match(r"([^\n<(]+)", block)
            if not m:
                continue
            name = _strip_ns(m.group(1).strip())
            args = {}
            for k, v in pairs:
                k, v = k.strip(), v.strip()
                t = _arg_type(name, k, tools)
                if t and t != "string":
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                args[k] = v
            if name:
                calls.append(_one_call(name, args))
            continue
        m = _TS_CALL.match(block)  # syntax 2: TS-style call with object-literal args
        if not m:
            continue
        name, args = _strip_ns(m.group(1).strip()), _parse_object_literal(m.group(2))
        if args is None or not name:
            continue
        if name == "parallel" and isinstance(args.get("tool_uses"), list):
            # multi_tool_use.parallel wrapper -> expand into individual calls
            for tu in args["tool_uses"]:
                tn = _strip_ns(str(tu.get("recipient_name", "")).strip())
                if tn:
                    calls.append(_one_call(tn, tu.get("parameters") or {}))
        else:
            calls.append(_one_call(name, args))
    return normal, calls
