'''Keep in the final mkvmerge only the fonts the kept ASS/SSA tracks use, each font once, and one cover image.'''

from hashlib import sha1
from json import loads
from os import path
from re import compile
from struct import unpack, error as struct_error

import tools

FONT_NAME_IDS = {1, 4, 6, 16}
STYLE_SECTIONS = {"[v4+ styles]", "[v4 styles]"}
FN_OVERRIDE_TAG = compile(r"\\fn([^\\}]*)")

def restrict_final_attachments(final_insert, master_path, metadata_cmd, split_path, split_subtitles):
    """Rewrite final_insert in place: each source keeps its fonts used by the kept ASS/SSA tracks, de-duplicated, and at most one cover image."""
    try:
        mkvextract = tools.software.get("mkvextract", "mkvextract")

        list_ass_text = []
        for subs in split_subtitles.values():
            for sub in subs:
                if sub["keep"] and sub["ffprobe"]["codec_name"].lower() in ("ass", "ssa"):
                    stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([mkvextract, "-q", split_path, "tracks", f"{sub['StreamOrder']}:/dev/stdout"], 2, 120)
                    list_ass_text.append(stdout.decode("utf-8", errors="ignore").lstrip("\ufeff"))
        set_font_reference = set()
        for ass_text in list_ass_text:
            """ Example:
            [V4+ Styles]
            Format: Name, Fontname, Fontsize
            Style: Default,@Arial,20
            [Events]
            Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\fnTimes New Roman}Hello
            gives {"arial", "times new roman"}
            """
            in_style_section = False
            fontname_index = None
            for line in ass_text.splitlines():
                line_stripped = line.strip()
                line_lowered = line_stripped.lower()
                if line_lowered.startswith("[") and line_lowered.endswith("]"):
                    in_style_section = line_lowered in STYLE_SECTIONS
                    fontname_index = None
                elif in_style_section and line_lowered.startswith("format:"):
                    list_format_field = [field.strip().lower() for field in line_stripped.split(":", 1)[1].split(",")]
                    fontname_index = None
                    if "fontname" in list_format_field:
                        fontname_index = list_format_field.index("fontname")
                elif in_style_section and line_lowered.startswith("style:") and fontname_index is not None:
                    list_style_field = line_stripped.split(":", 1)[1].split(",")
                    if fontname_index < len(list_style_field):
                        set_font_reference.add(list_style_field[fontname_index].strip().lstrip("@").strip().lower())
            for fn_override in FN_OVERRIDE_TAG.finditer(ass_text):
                set_font_reference.add(fn_override.group(1).strip().lstrip("@").strip().lower())
        set_font_reference.discard("")

        list_source_file = [master_path]
        for cmd_part in metadata_cmd:
            if (not cmd_part.startswith("-")) and path.isfile(cmd_part):
                list_source_file.append(cmd_part)

        dict_attachments_by_source = {}
        for source_file in list_source_file:
            stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["mkvmerge"], "-J", source_file], 2, 120)
            dict_attachments_by_source[source_file] = loads(stdout.decode("utf-8")).get("attachments", [])

        cover_source = None
        cover_id = None
        cover_name = None
        for source_file in list_source_file:
            list_image = sorted([(attachment.get("content_type", ""), attachment["file_name"], attachment["id"]) for attachment in dict_attachments_by_source[source_file] if attachment.get("content_type", "").startswith("image/")])
            if list_image and cover_source is None:
                cover_source = source_file
                cover_name = list_image[0][1]
                cover_id = list_image[0][2]

        set_kept_font = set()
        set_font_hash = set()
        for source_file in list_source_file:
            for attachment in dict_attachments_by_source[source_file]:
                content_type = attachment.get("content_type", "")
                if (not content_type.startswith("image/")) and (content_type.startswith("font/") or attachment["file_name"].lower().endswith((".ttf", ".otf", ".ttc"))):
                    font_data, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([mkvextract, "-q", source_file, "attachments", f"{attachment['id']}:/dev/stdout"], 2, 120)
                    set_font_name = set()
                    try:
                        list_sfnt_offset = [0]
                        if font_data[:4] == b"ttcf":
                            number_font = unpack(">I", font_data[8:12])[0]
                            list_sfnt_offset = list(unpack(f">{number_font}I", font_data[12:12 + 4 * number_font]))
                        for sfnt_offset in list_sfnt_offset:
                            number_table = unpack(">H", font_data[sfnt_offset + 4:sfnt_offset + 6])[0]
                            name_table_offset = None
                            for table_position in range(number_table):
                                table_entry_offset = sfnt_offset + 12 + 16 * table_position
                                if name_table_offset is None and font_data[table_entry_offset:table_entry_offset + 4] == b"name":
                                    name_table_offset = unpack(">I", font_data[table_entry_offset + 8:table_entry_offset + 12])[0]
                            if name_table_offset is not None:
                                table_format, number_record, string_offset = unpack(">HHH", font_data[name_table_offset:name_table_offset + 6])
                                for record_position in range(number_record):
                                    record_offset = name_table_offset + 6 + 12 * record_position
                                    platform_id, encoding_id, language_id, name_id, name_length, name_offset = unpack(">HHHHHH", font_data[record_offset:record_offset + 12])
                                    if name_id in FONT_NAME_IDS:
                                        name_start = name_table_offset + string_offset + name_offset
                                        name_encoding = "mac_roman" if platform_id == 1 else "utf-16-be"
                                        set_font_name.add(font_data[name_start:name_start + name_length].decode(name_encoding, errors="ignore").strip().lower())
                    except (struct_error, IndexError):
                        set_font_name = set()
                    set_font_name.discard("")
                    font_hash = sha1(font_data).hexdigest()
                    if (set_font_name & set_font_reference) and font_hash not in set_font_hash:
                        set_font_hash.add(font_hash)
                        set_kept_font.add((source_file, attachment["id"]))

        dict_kept_ids_by_source = {}
        number_attachment = 0
        number_kept = 0
        for source_file in list_source_file:
            list_kept_id = [attachment["id"] for attachment in dict_attachments_by_source[source_file] if (source_file, attachment["id"]) in set_kept_font or (source_file == cover_source and attachment["id"] == cover_id)]
            dict_kept_ids_by_source[source_file] = list_kept_id
            number_attachment += len(dict_attachments_by_source[source_file])
            number_kept += len(list_kept_id)

        list_source_position = []
        for source_file in list_source_file:
            number_match = final_insert.count(source_file)
            if number_match == 1:
                list_source_position.append((final_insert.index(source_file), source_file))
            elif tools.dev:
                tools.logs.append(f"\t\tattachments: skipped {source_file}, {number_match} matches in the command\n")
        for source_position, source_file in sorted(list_source_position, reverse=True):
            if dict_kept_ids_by_source[source_file]:
                final_insert[source_position:source_position] = ["--attachments", ",".join([str(attachment_id) for attachment_id in dict_kept_ids_by_source[source_file]])]
            else:
                final_insert.insert(source_position, "-M")

        tools.logs.append(f"\t\tattachments: kept {number_kept}/{number_attachment} refs={len(set_font_reference)} cover={cover_name}\n")
        if tools.dev:
            for source_file, list_kept_id in dict_kept_ids_by_source.items():
                tools.logs.append(f"\t\tattachments: {source_file} kept {list_kept_id}\n")
    except Exception as e:
        tools.logs.append(f"\t\tattachments: failed, kept everything ({e})\n")
