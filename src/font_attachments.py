'''Keep each font attachment once, only when a kept ASS/SSA track needs it, and no other attachment.'''

from hashlib import sha1
from json import loads
from re import compile
from struct import unpack, error as struct_error

import tools
import video

FONT_NAME_IDS = {1, 4, 6, 16}
FONT_CONTENT_TYPES = ("font/", "application/x-truetype-font", "application/vnd.ms-opentype", "application/font-sfnt")
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")
STYLE_SECTIONS = {"[v4+ styles]", "[v4 styles]"}
FN_OVERRIDE_TAG = compile(r"\\fn([^\\}]*)")

def list_fonts_needed(video_obj, sub_track_to_convert_or_keep, font_tracker):
    """Add to font_tracker["font_needed"] the lowercase font names the kept ASS/SSA tracks of video_obj reference."""
    for sub in sub_track_to_convert_or_keep:
        if video.get_codec(sub) in ("ass", "ssa"):
            try:
                stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["mkvextract"], "-q", video_obj.filePath, "tracks", f"{sub['StreamOrder']}:/dev/stdout"], 2, 120)
                ass_text = stdout.decode("utf-8", errors="ignore").lstrip("﻿")
            except Exception as e:
                tools.log_always(f"[fonts WARN] subtitle {sub['StreamOrder']} of {video_obj.filePath} not read, every font of the file counts as needed: {e}\n")
                font_tracker["file_all_fonts_needed"].add(video_obj.filePath)
                ass_text = ""
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
                        font_tracker["font_needed"].add(list_style_field[fontname_index].strip().lstrip("@").strip().lower())
            for fn_override in FN_OVERRIDE_TAG.finditer(ass_text):
                font_tracker["font_needed"].add(fn_override.group(1).strip().lstrip("@").strip().lower())
    font_tracker["font_needed"].discard("")

def get_attachments_to_keep(file_path, font_tracker):
    """Return the mkvmerge option that keeps, from file_path, only the fonts not already added and needed so far: [] keeps all."""
    param_to_send = []
    try:
        stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["mkvmerge"], "-J", file_path], 2, 120)
        list_attachment = loads(stdout.decode("utf-8")).get("attachments", [])
        list_kept_id = []
        list_dropped_name = []
        for attachment in list_attachment:
            if not (attachment.get("content_type", "").startswith(FONT_CONTENT_TYPES) or attachment["file_name"].lower().endswith(FONT_EXTENSIONS)):
                list_dropped_name.append(attachment["file_name"])
            else:
                font_uid = (attachment.get("properties") or {}).get("uid")
                if font_uid is None or font_uid not in font_tracker["font_hash_by_uid"]:
                    font_data, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["mkvextract"], "-q", file_path, "attachments", f"{attachment['id']}:/dev/stdout"], 2, 120)
                    font_hash = sha1(font_data).hexdigest()
                    if font_uid is not None:
                        font_tracker["font_hash_by_uid"][font_uid] = font_hash
                else:
                    font_hash = font_tracker["font_hash_by_uid"][font_uid]
                if font_hash not in font_tracker["font_names_by_hash"]:
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
                    font_tracker["font_names_by_hash"][font_hash] = frozenset(set_font_name)
                font_names = font_tracker["font_names_by_hash"][font_hash]
                if file_path in font_tracker["file_all_fonts_needed"]:
                    font_tracker["font_needed"].update(font_names)
                if font_hash in font_tracker["font_hash_added"] or (font_names and (font_names in font_tracker["font_names_added"] or not (font_names & font_tracker["font_needed"]))):
                    list_dropped_name.append(attachment["file_name"])
                else:
                    font_tracker["font_hash_added"].add(font_hash)
                    font_tracker["font_names_added"].add(font_names)
                    list_kept_id.append(attachment["id"])
        if not list_kept_id:
            param_to_send = ["-M"]
        elif list_dropped_name:
            param_to_send = ["--attachments", ",".join([str(attachment_id) for attachment_id in list_kept_id])]
        if tools.dev:
            tools.logs.append(f"\t\tattachments of {file_path}: kept {len(list_kept_id)}/{len(list_attachment)}, dropped {list_dropped_name}\n")
    except Exception as e:
        tools.log_always(f"[fonts WARN] attachments of {file_path} all kept: {e}\n")
        param_to_send = []
    return param_to_send
