import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


CONTROL_KEYWORDS = ("if", "for", "try", "catch")
KEYWORDS = {
    "if", "for", "while", "switch", "case", "default", "do", "else",
    "try", "catch", "finally", "throw", "throws", "return", "new",
    "class", "interface", "enum", "extends", "implements", "this", "super",
    "synchronized", "assert", "break", "continue",
}

COMMON_API_PREFIXES = (
    "java.",
    "javax.",
    "org.apache.",
    "org.slf4j.",
    "com.google.",
    "com.fasterxml.",
    "org.junit.",
)

COMMON_API_ROOTS = {
    "System", "Math", "Collections", "Arrays", "Objects", "Files", "Paths",
    "Optional", "Stream", "Collectors", "String", "Integer", "Long",
    "Double", "Float", "Boolean", "Character", "BigInteger", "BigDecimal",
    "Logger", "Log",
}

LOG_KEYWORDS = {
    "log", "logger", "logging", "trace", "debug", "info", "warn", "warning",
    "error", "fatal",
}

METHOD_DECL_RE = re.compile(
    r"""
    (?P<mods>(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default)\s+)*
    (?P<ret>[A-Za-z_][\w\.\<\>\[\],\s\?]*?)\s+
    (?P<name>[A-Za-z_]\w*)\s*
    \((?P<params>[^)]*)\)
    (?:\s*throws\s+(?P<throws>[^{]+))?
    \s*\{
    """,
    re.VERBOSE | re.MULTILINE,
)

PACKAGE_DECL_RE = re.compile(r"\bpackage\s+([A-Za-z_][\w\.]*)\s*;")
TYPE_DECL_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_]\w*)")


def mask_comments_and_strings(code: str) -> str:
    result = list(code)
    i = 0
    in_line = False
    in_block = False
    in_string = False
    in_char = False
    escape = False
    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""
        if in_line:
            if ch == "\n":
                in_line = False
            else:
                result[i] = " "
        elif in_block:
            if ch == "*" and nxt == "/":
                result[i] = " "
                result[i + 1] = " "
                i += 1
                in_block = False
            else:
                result[i] = " "
        elif in_string:
            result[i] = " "
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
        elif in_char:
            result[i] = " "
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_char = False
        else:
            if ch == "/" and nxt == "/":
                result[i] = " "
                result[i + 1] = " "
                i += 1
                in_line = True
            elif ch == "/" and nxt == "*":
                result[i] = " "
                result[i + 1] = " "
                i += 1
                in_block = True
            elif ch == "\"":
                result[i] = " "
                in_string = True
            elif ch == "'":
                result[i] = " "
                in_char = True
        i += 1
    return "".join(result)


def find_matching_brace(masked: str, start_idx: int) -> int:
    depth = 0
    i = start_idx
    while i < len(masked):
        ch = masked[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_camel(token: str) -> List[str]:
    if not token:
        return []
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+", token)
    return [p.lower() for p in parts if p]


def normalize_identifier(name: str) -> List[str]:
    if not name:
        return []
    name = re.sub(r"<[^>]*>", " ", name)
    name = name.replace("[]", " ")
    parts = re.split(r"[^A-Za-z0-9]+", name)
    tokens: List[str] = []
    for part in parts:
        tokens.extend(split_camel(part))
    return [t for t in tokens if t]


def extract_string_literals(code: str) -> List[str]:
    literals: List[str] = []
    i = 0
    in_line = False
    in_block = False
    in_string = False
    escape = False
    buf: List[str] = []
    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""
        if in_line:
            if ch == "\n":
                in_line = False
        elif in_block:
            if ch == "*" and nxt == "/":
                i += 1
                in_block = False
        elif in_string:
            if escape:
                buf.append(ch)
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                literals.append("".join(buf))
                buf = []
                in_string = False
            else:
                buf.append(ch)
        else:
            if ch == "/" and nxt == "/":
                in_line = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block = True
                i += 1
            elif ch == "\"":
                in_string = True
        i += 1
    return literals


def extract_signature(method_code: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    masked = mask_comments_and_strings(method_code)
    match = METHOD_DECL_RE.search(masked)
    if not match:
        return [], [], [], []
    ret = match.group("ret") or ""
    name = match.group("name") or ""
    params = match.group("params") or ""
    throws = match.group("throws") or ""
    return (
        normalize_identifier(ret),
        normalize_identifier(name),
        extract_param_names(params),
        extract_throws_types(throws),
    )


def extract_param_names(params: str) -> List[str]:
    if not params.strip():
        return []
    tokens: List[str] = []
    for param in params.split(","):
        param = param.strip()
        if not param:
            continue
        param = re.sub(r"@[\w\.]+", "", param).strip()
        parts = param.split()
        if not parts:
            continue
        name = parts[-1].replace("...", "")
        tokens.extend(normalize_identifier(name))
    return tokens


def extract_throws_types(throws: str) -> List[str]:
    if not throws.strip():
        return []
    types: List[str] = []
    for item in throws.split(","):
        item = item.strip()
        if not item:
            continue
        item = re.sub(r"<[^>]*>", "", item)
        types.extend(normalize_identifier(item.split()[-1]))
    return types


def extract_control_tokens(masked_body: str) -> List[str]:
    tokens: List[str] = []
    for m in re.finditer(r"\b(if|for|try|catch)\b", masked_body):
        tokens.append(m.group(1))
    return tokens


def is_common_api(qual: str) -> bool:
    qual = qual.strip(".")
    if any(qual.startswith(prefix) for prefix in COMMON_API_PREFIXES):
        return True
    root = qual.split(".")[0]
    return root in COMMON_API_ROOTS


def extract_call_tokens(masked_body: str) -> List[str]:
    tokens: List[str] = []
    for m in re.finditer(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", masked_body):
        full = m.group(1)
        base = full.split(".")[-1]
        if base in KEYWORDS or full in KEYWORDS:
            continue
        tokens.extend(normalize_identifier(base))
        if is_common_api(full):
            tokens.extend(normalize_identifier(full))
    return tokens


def extract_exception_types(masked_body: str) -> List[str]:
    types: List[str] = []
    for m in re.finditer(r"\bcatch\s*\(\s*([A-Za-z_][\w\.\<\>|\s]*)", masked_body):
        t = m.group(1)
        for part in t.split("|"):
            part = part.strip()
            if part:
                types.extend(normalize_identifier(part))
    for m in re.finditer(r"\bthrow\s+new\s+([A-Za-z_][\w\.\<\>]*)", masked_body):
        types.extend(normalize_identifier(m.group(1)))
    return types


def extract_constants(masked_body: str, string_literals: List[str]) -> List[str]:
    tokens: List[str] = []
    for m in re.finditer(r"\b\d+(?:\.\d+)?(?:[lLfFdD])?\b", masked_body):
        val = m.group(0)
        if len(val) > 6:
            tokens.append("num")
        else:
            tokens.append(val)
    for lit in string_literals:
        if not lit:
            continue
        words = re.findall(r"[A-Za-z0-9]+", lit)
        if words:
            for w in words[:3]:
                tokens.extend(normalize_identifier(w))
        else:
            tokens.append("str")
    for m in re.finditer(r"\b(true|false|null)\b", masked_body):
        tokens.append(m.group(1))
    return tokens


def extract_log_keywords(masked_body: str, string_literals: List[str]) -> List[str]:
    tokens: List[str] = []
    for m in re.finditer(r"\b(" + "|".join(sorted(LOG_KEYWORDS)) + r")\b", masked_body, flags=re.IGNORECASE):
        tokens.append(m.group(1).lower())
    for lit in string_literals:
        for w in re.findall(r"[A-Za-z]+", lit):
            lw = w.lower()
            if lw in LOG_KEYWORDS:
                tokens.append(lw)
    return tokens


def compact_method(method_code: str, max_tokens: int = 256) -> str:
    masked = mask_comments_and_strings(method_code)
    brace_idx = masked.find("{")
    body = method_code[brace_idx + 1:] if brace_idx != -1 else method_code
    masked_body = masked[brace_idx + 1:] if brace_idx != -1 else masked

    ret_tokens, name_tokens, param_tokens, throws_tokens = extract_signature(method_code)
    sig_tokens = ["#sig"] + ret_tokens + name_tokens + param_tokens + throws_tokens

    control_tokens = ["#ctrl"] + extract_control_tokens(masked_body)
    call_tokens = ["#call"] + extract_call_tokens(masked_body)

    string_literals = extract_string_literals(body)
    const_tokens = ["#const"] + extract_constants(masked_body, string_literals)
    exc_tokens = ["#exc"] + extract_exception_types(masked_body)
    log_tokens = ["#log"] + extract_log_keywords(masked_body, string_literals)

    section_limits = {
        "sig": 32,
        "ctrl": 24,
        "call": 96,
        "const": 24,
        "exc": 24,
        "log": 16,
    }

    sig_tokens = sig_tokens[: section_limits["sig"]]
    control_tokens = control_tokens[: section_limits["ctrl"]]
    call_tokens = call_tokens[: section_limits["call"]]
    const_tokens = const_tokens[: section_limits["const"]]
    exc_tokens = exc_tokens[: section_limits["exc"]]
    log_tokens = log_tokens[: section_limits["log"]]

    seq = sig_tokens + control_tokens + call_tokens + const_tokens + exc_tokens + log_tokens
    if len(seq) > max_tokens:
        seq = seq[:max_tokens]
    return " ".join(seq)


def _extract_package(code: str) -> str:
    match = PACKAGE_DECL_RE.search(code)
    return match.group(1) if match else ""


def _type_decls(masked: str) -> List[Tuple[int, str]]:
    decls: List[Tuple[int, str]] = []
    for m in TYPE_DECL_RE.finditer(masked):
        decls.append((m.start(), m.group(2)))
    return decls


def _nearest_type_name(decls: List[Tuple[int, str]], pos: int) -> str:
    name = ""
    for start, cls_name in decls:
        if start > pos:
            break
        name = cls_name
    return name


def _build_key(pkg: str, cls: str, method: str) -> str:
    if cls:
        qual = f"{pkg}.{cls}" if pkg else cls
        return f"{qual}:{method}"
    return method


def iter_methods(code: str) -> Iterable[Dict[str, object]]:
    masked = mask_comments_and_strings(code)
    package_name = _extract_package(masked)
    type_decls = _type_decls(masked)
    matches = list(METHOD_DECL_RE.finditer(masked))
    for m in matches:
        brace_idx = masked.find("{", m.end() - 1)
        if brace_idx == -1:
            continue
        end_idx = find_matching_brace(masked, brace_idx)
        if end_idx == -1:
            continue
        method_name = m.group("name") or ""
        class_name = _nearest_type_name(type_decls, m.start())
        method_code = code[m.start(): end_idx + 1]
        start_line = code.count("\n", 0, m.start()) + 1
        end_line = code.count("\n", 0, end_idx) + 1
        yield {
            "name": method_name,
            "class": class_name,
            "package": package_name,
            "key": _build_key(package_name, class_name, method_name),
            "start": m.start(),
            "end": end_idx + 1,
            "start_line": start_line,
            "end_line": end_line,
            "code": method_code,
        }

    if type_decls:
        for _, cls in type_decls:
            ctor_re = re.compile(
                r"\b(?:public|protected|private)\s+"
                + re.escape(cls)
                + r"\s*\((?P<params>[^)]*)\)\s*\{",
                re.MULTILINE,
            )
            for m in ctor_re.finditer(masked):
                brace_idx = masked.find("{", m.end() - 1)
                if brace_idx == -1:
                    continue
                end_idx = find_matching_brace(masked, brace_idx)
                if end_idx == -1:
                    continue
                method_code = code[m.start(): end_idx + 1]
                start_line = code.count("\n", 0, m.start()) + 1
                end_line = code.count("\n", 0, end_idx) + 1
                yield {
                    "name": cls,
                    "class": cls,
                    "package": package_name,
                    "key": _build_key(package_name, cls, cls),
                    "start": m.start(),
                    "end": end_idx + 1,
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": method_code,
                }


def iter_java_files(root: str) -> Iterable[str]:
    if os.path.isfile(root) and root.lower().endswith(".java"):
        yield root
        return
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".java"):
                yield os.path.join(dirpath, name)


def encode_with_codebert(seqs: List[str], model_path: str, max_length: int = 256, batch_size: int = 8) -> List[List[float]]:
    import torch
    from transformers import RobertaModel, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings: List[List[float]] = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().tolist()
            embeddings.extend(cls)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact Java methods into short semantic sequences for CodeBERT.")
    parser.add_argument("--input", required=True, help="Path to a Java file or a source root.")
    parser.add_argument("--output", required=True, help="Output JSONL path for compacted sequences.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max token count for compacted sequence.")
    parser.add_argument("--model-path", default=None, help="Optional CodeBERT model path for embeddings.")
    parser.add_argument("--embed-output", default=None, help="Output path for embeddings JSONL.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length for CodeBERT.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for CodeBERT encoding.")
    args = parser.parse_args()

    records: List[Dict[str, object]] = []
    for path in iter_java_files(args.input):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        for method in iter_methods(code):
            compact = compact_method(method["code"], max_tokens=args.max_tokens)
            record = {
                "file": path,
                "method": method["name"],
                "class": method["class"],
                "package": method["package"],
                "key": method["key"],
                "start_line": method["start_line"],
                "end_line": method["end_line"],
                "compact": compact,
            }
            records.append(record)

    with open(args.output, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=True) + "\n")

    if args.model_path:
        embed_path = args.embed_output or os.path.splitext(args.output)[0] + ".embeddings.jsonl"
        seqs = [r["compact"] for r in records]
        embeddings = encode_with_codebert(seqs, args.model_path, max_length=args.max_length, batch_size=args.batch_size)
        with open(embed_path, "w", encoding="utf-8") as out:
            for rec, emb in zip(records, embeddings):
                out.write(json.dumps({"file": rec["file"], "method": rec["method"], "key": rec["key"], "embedding": emb}, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
