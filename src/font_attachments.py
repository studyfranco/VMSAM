'''
Created on 12 Oct 2026

@author: studyfranco
'''
import hashlib
import json
import os
import re
import struct
import tools

NAME_IDS_WANTED = {1, 4, 6, 16}
STYLE_SECTIONS = {"[v4+ styles]", "[v4 styles]"}
FN_TAG = re.compile(r"\\fn([^\\}]*)")

def sfnt_offsets(data):
    tag = data[:4]
    if tag == b"ttcf":
        num_fonts = struct.unpack(">I", data[8:12])[0]
        return list(struct.unpack(f">{num_fonts}I", data[12:12 + 4 * num_fonts]))
    return [0]

def name_table_strings(data, sfnt_offset):
    num_tables = struct.unpack(">H", data[sfnt_offset + 4:sfnt_offset + 6])[0]
    directory = data[sfnt_offset + 12:sfnt_offset + 12 + 16 * num_tables]
    name_table_offset = None
    for i in range(num_tables):
        entry = directory[i * 16:(i + 1) * 16]
        if entry[:4] == b"name":
            name_table_offset = struct.unpack(">I", entry[8:12])[0]
            break
    if name_table_offset is None:
        return set()
    _, count, string_offset = struct.unpack(">HHH", data[name_table_offset:name_table_offset + 6])
    strings_base = name_table_offset + string_offset
    names = set()
    for i in range(count):
        record_offset = name_table_offset + 6 + i * 12
        platform_id, _, _, name_id, length, offset = struct.unpack(">HHHHHH", data[record_offset:record_offset + 12])
        if name_id not in NAME_IDS_WANTED:
            continue
        raw = data[strings_base + offset:strings_base + offset + length]
        encoding = "mac_roman" if platform_id == 1 else "utf-16-be"
        text = raw.decode(encoding, errors="ignore").strip().lower()
        if text:
            names.add(text)
    return names

def sfnt_names(data: bytes) -> set:
    """Read the family/full/postscript/typographic names out of a TTF, OTF or TTC blob.

    >>> sfnt_names(b"garbage") == set()
    True
    """
    names = set()
    try:
        for offset in sfnt_offsets(data):
            names |= name_table_strings(data, offset)
    except (struct.error, IndexError):
        return set()
    return names

def normalize_font_name(name: str) -> str:
    return name.strip().lstrip("@").strip().lower()

def style_fontname_index(format_line):
    fields = [field.strip().lower() for field in format_line.split(":", 1)[1].split(",")]
    return fields.index("fontname") if "fontname" in fields else None

def ass_font_refs(text: str) -> set:
    """Collect every font name a kept ASS/SSA track references, from Style lines and \\fn overrides.

    >>> ass_font_refs("[V4+ Styles]\\nFormat: Name, Fontname\\nStyle: Default,Arial\\n") == {"arial"}
    True
    """
    text = text.lstrip("﻿")
    refs = set()
    in_style_section = False
    fontname_index = None
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("[") and lowered.endswith("]"):
            in_style_section = lowered in STYLE_SECTIONS
            fontname_index = None
            continue
        if not in_style_section:
            continue
        if lowered.startswith("format:"):
            fontname_index = style_fontname_index(stripped)
        elif lowered.startswith("style:") and fontname_index is not None:
            fields = stripped.split(":", 1)[1].split(",")
            if fontname_index < len(fields):
                name = normalize_font_name(fields[fontname_index])
                if name:
                    refs.add(name)
    for match in FN_TAG.finditer(text):
        name = normalize_font_name(match.group(1))
        if name:
            refs.add(name)
    return refs

def mkvmerge_json(file_path):
    stdout, _, _ = tools.launch_cmdExt_with_timeout_reload([tools.software["mkvmerge"], "-J", file_path], 2, 120)
    return json.loads(stdout.decode("utf-8"))

def extract_attachments(file_path, attachments, out_dir):
    if not attachments:
        return
    mkvextract = tools.software.get("mkvextract", "mkvextract")
    spec = [f"{a['id']}:{os.path.join(out_dir, f"{a['id']}_{a['file_name']}")}" for a in attachments]
    tools.launch_cmdExt_with_timeout_reload([mkvextract, file_path, "attachments"] + spec, 2, 120)

def extracted_path(out_dir, attachment):
    return os.path.join(out_dir, f"{attachment['id']}_{attachment['file_name']}")

def is_font_attachment(attachment):
    content_type = attachment.get("content_type", "")
    return content_type.startswith("font/") or attachment["file_name"].lower().endswith((".ttf", ".otf", ".ttc"))

def is_image_attachment(attachment):
    return attachment.get("content_type", "").startswith("image/")

def source_paths(master_path, metadata_cmd):
    paths = [master_path]
    for token in metadata_cmd:
        if (not token.startswith("-")) and os.path.isfile(token):
            paths.append(token)
    return paths

def kept_ass_texts(split_path, split_subtitles, out_dir):
    kept = []
    for subs in split_subtitles.values():
        for sub in subs:
            if sub["keep"] and sub["ffprobe"]["codec_name"].lower() in ("ass", "ssa"):
                kept.append(sub)
    if not kept:
        return []
    mkvextract = tools.software.get("mkvextract", "mkvextract")
    spec = [f"{sub['StreamOrder']}:{os.path.join(out_dir, f"{sub['StreamOrder']}.ass")}" for sub in kept]
    tools.launch_cmdExt_with_timeout_reload([mkvextract, split_path, "tracks"] + spec, 2, 120)
    texts = []
    for sub in kept:
        extracted = os.path.join(out_dir, f"{sub['StreamOrder']}.ass")
        with open(extracted, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return texts

def ass_references_in_kept_tracks(split_path, split_subtitles, out_dir):
    refs = set()
    for text in kept_ass_texts(split_path, split_subtitles, out_dir):
        refs |= ass_font_refs(text)
    return refs

def keep_referenced_font(attachment, out_dir, refs, seen_hashes):
    with open(extracted_path(out_dir, attachment), "rb") as f:
        data = f.read()
    if not sfnt_names(data) & refs:
        return False
    content_hash = hashlib.sha1(data).hexdigest()
    if content_hash in seen_hashes:
        return False
    seen_hashes.add(content_hash)
    return True

def cover_sort_key(attachment):
    return (attachment.get("content_type", ""), attachment["file_name"])

def choose_cover(attachments_by_source, sources):
    """Pick one cover image to keep: the master's first by (mimetype, name) if it has any, else the first candidate's.

    >>> choose_cover({"m": [], "c": [{"id": 3, "content_type": "image/jpeg", "file_name": "cover.jpg"}]}, ["m", "c"])
    ('c', 3)
    """
    for source in sources:
        images = sorted((a for a in attachments_by_source[source] if is_image_attachment(a)), key=cover_sort_key)
        if images:
            return source, images[0]["id"]
    return None, None

def cover_file_name(attachments_by_source, cover_source, cover_id):
    if cover_source is None:
        return None
    for attachment in attachments_by_source[cover_source]:
        if attachment["id"] == cover_id:
            return attachment["file_name"]
    return None

def select_attachment_ids(attachments, source, refs, seen_hashes, out_dir, cover_source, cover_id):
    kept_ids = []
    for attachment in attachments:
        if is_image_attachment(attachment):
            if source == cover_source and attachment["id"] == cover_id:
                kept_ids.append(attachment["id"])
            continue
        if not is_font_attachment(attachment):
            continue
        if keep_referenced_font(attachment, out_dir, refs, seen_hashes):
            kept_ids.append(attachment["id"])
    return kept_ids

def insert_attachment_selectors(final_insert, sources, kept_ids_by_source):
    indices = []
    for source in sources:
        matches = [i for i, token in enumerate(final_insert) if token == source]
        if len(matches) != 1:
            if tools.dev:
                tools.logs.append(f"\t\tattachments: skipped {source}, {len(matches)} matches in the command\n")
            continue
        indices.append((matches[0], source))
    for index, source in sorted(indices, reverse=True):
        kept_ids = kept_ids_by_source[source]
        selector = ["--attachments", ",".join(str(i) for i in kept_ids)] if kept_ids else ["-M"]
        final_insert[index:index] = selector

def apply_attachments_restriction(final_insert, master_path, metadata_cmd, split_path, split_subtitles):
    sources = source_paths(master_path, metadata_cmd)
    extract_dir = os.path.join(tools.tmpFolder, f"font_attachments_{hashlib.md5(split_path.encode()).hexdigest()[:16]}")
    tools.make_dirs(extract_dir)
    try:
        refs = ass_references_in_kept_tracks(split_path, split_subtitles, extract_dir)
        attachments_by_source = {}
        for source in sources:
            attachments = mkvmerge_json(source).get("attachments", [])
            attachments_by_source[source] = attachments
            extract_attachments(source, attachments, extract_dir)
        cover_source, cover_id = choose_cover(attachments_by_source, sources)
        seen_hashes = set()
        kept_ids_by_source = {}
        total_attachments = 0
        kept_count = 0
        for source in sources:
            attachments = attachments_by_source[source]
            total_attachments += len(attachments)
            kept_ids = select_attachment_ids(attachments, source, refs, seen_hashes, extract_dir, cover_source, cover_id)
            kept_ids_by_source[source] = kept_ids
            kept_count += len(kept_ids)
        insert_attachment_selectors(final_insert, sources, kept_ids_by_source)
        cover_name = cover_file_name(attachments_by_source, cover_source, cover_id)
        tools.logs.append(f"\t\tattachments: kept {kept_count}/{total_attachments} refs={len(refs)} cover={cover_name}\n")
        if tools.dev:
            for source, kept_ids in kept_ids_by_source.items():
                tools.logs.append(f"\t\tattachments: {source} kept {kept_ids}\n")
    finally:
        tools.remove_dir(extract_dir, printError=False)

def restrict_final_attachments(final_insert, master_path, metadata_cmd, split_path, split_subtitles):
    """Rewrite final_insert in place: each source keeps only its referenced, de-duplicated fonts, plus at most one cover image.

    Fails open: any exception leaves final_insert untouched and logs one line.
    """
    try:
        apply_attachments_restriction(final_insert, master_path, metadata_cmd, split_path, split_subtitles)
    except Exception as e:
        tools.logs.append(f"\t\tattachments: failed, kept everything ({e})\n")
