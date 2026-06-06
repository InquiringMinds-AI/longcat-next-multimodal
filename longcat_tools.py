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


# ---- output parsing: <longcat_tool_call> XML -> OpenAI tool_calls ----
_TC = re.compile(r"<longcat_tool_call>(.*?)</longcat_tool_call>", re.DOTALL)
_PAIR = re.compile(r"<longcat_arg_key>(.*?)</longcat_arg_key>\s*<longcat_arg_value>(.*?)</longcat_arg_value>", re.DOTALL)


def _arg_type(name, key, tools):
    for t in tools or []:
        fn = t.get("function") if t.get("type") == "function" else t
        if fn and fn.get("name") == name:
            return fn.get("parameters", {}).get("properties", {}).get(key, {}).get("type")
    return None


def parse_tool_calls(text, tools):
    """Return (normal_text, tool_calls[]). tool_calls in OpenAI shape (arguments = JSON string)."""
    if "<longcat_tool_call>" not in text:
        return text, []
    idx = text.find("<longcat_tool_call>")
    normal = text[:idx].strip()
    calls = []
    for block in _TC.findall(text):
        m = re.match(r"([^\n<]+)", block.strip())
        if not m:
            continue
        name = m.group(1).strip()
        # model calls via the TS namespace ("functions.get_weather"); OpenAI clients expect the
        # bare declared name -> strip a leading "functions." / "multi_tool_use." namespace prefix.
        for ns in ("functions.", "multi_tool_use."):
            if name.startswith(ns):
                name = name[len(ns):]
        args = {}
        for k, v in _PAIR.findall(block):
            k, v = k.strip(), v.strip()
            t = _arg_type(name, k, tools)
            if t and t != "string":
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            args[k] = v
        if name:
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return normal, calls
