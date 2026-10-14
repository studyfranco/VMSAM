// VMSAM web interface — vanilla JS, no external dependency.
//
// Sections:
//   1. utilities (toast, toggles)
//   2. API client
//   3. filename analysis engine (pure functions, no DOM)
//   4. tabs (section switch, header title link, home tiles)
//   5. folder creation
//   6. shared widgets: folder picker, existing-rules panel, directory picker,
//      file modal
//   7. group board (regex / index / incrementaller cards, drag & drop)
//   8. Shows tabs: bulk regex, index folder, move folder, incrementaller,
//      episodes, folder settings, rules, errors (Movies tabs: movie.js)

// --- 1. Utilities ---

function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' :
        type === 'success' ? 'toast-success' :
            '' // Default is info
        }`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.classList.add('visible');
    });

    setTimeout(() => {
        toast.classList.remove('visible');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function copyPattern(pattern) {
    navigator.clipboard.writeText(pattern);
    showToast('Pattern copied to clipboard!', 'success');
}

function toggleCollapsible(contentId, iconId) {
    const content = document.getElementById(contentId);
    const icon = document.getElementById(iconId);
    if (!content) return;
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        if (icon) icon.textContent = '▲';
    } else {
        content.classList.add('hidden');
        if (icon) icon.textContent = '▼';
    }
}

function toggleHelp() {
    toggleCollapsible('help-content', 'help-icon');
}

function toggleAdvanced() {
    const content = document.getElementById('advanced-options');
    const icon = document.getElementById('adv-icon');
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        icon.textContent = '▼';
    } else {
        content.classList.add('hidden');
        icon.textContent = '▶';
    }
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
}

// In-page replacement of window.confirm(), themed. Resolves true (OK) or
// false (Cancel, Escape, a click outside the box). OK has the focus, so
// Enter = OK; Escape = Cancel; Tab cycles between the two buttons only (Enter
// on a focused Cancel cancels). The message is plain text (file names carry
// brackets), its line breaks are kept.
function confirmDialog(message, options = {}) {
    return new Promise(resolve => {
        const previousFocus = document.activeElement;
        const overlay = el('div', 'modal-overlay confirm-overlay');
        const box = el('div', 'modal-box confirm-box');
        box.setAttribute('role', 'alertdialog');
        box.setAttribute('aria-modal', 'true');
        const title = el('h3', 'confirm-title', options.title || 'Please confirm');
        title.id = 'confirm-dialog-title';
        box.setAttribute('aria-labelledby', title.id);
        const text = el('div', 'confirm-message', message);
        const buttons = el('div', 'confirm-buttons');
        const cancel = el('button', 'btn btn-secondary confirm-cancel', options.cancelText || 'Cancel');
        const ok = el('button', 'btn btn-primary confirm-ok', options.okText || 'OK');
        cancel.type = 'button';
        ok.type = 'button';
        buttons.append(cancel, ok);
        box.append(title, text, buttons);
        overlay.appendChild(box);

        const close = (answer) => {
            document.removeEventListener('keydown', onKey, true);
            overlay.remove();
            if (previousFocus && previousFocus.focus) previousFocus.focus();
            resolve(answer);
        };
        const onKey = (e) => {
            if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(false); }
            else if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); close(document.activeElement !== cancel); }
            else if (e.key === 'Tab') { e.preventDefault(); (document.activeElement === ok ? cancel : ok).focus(); }
        };
        ok.onclick = () => close(true);
        cancel.onclick = () => close(false);
        overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(false); });
        document.addEventListener('keydown', onKey, true);
        document.body.appendChild(overlay);
        ok.focus();
    });
}

const VIDEO_EXTS = ['mkv', 'mp4', 'avi', 'm4v', 'mov', 'ts', 'webm', 'wmv'];

function isVideoName(name) {
    const lower = name.toLowerCase();
    return VIDEO_EXTS.some(ext => lower.endsWith('.' + ext));
}

// State
let state = {
    currentBaseDir: null,
    selectedFolderId: null,
    selectedFolderPath: null,
    regexCards: [], // legacy, kept for compatibility
    files: []
};

// --- 2. API Client ---

// Every proxied VMSAM call relays the upstream JSON body unchanged, so the
// `detail` of a 4xx is the message worth showing, not "request failed".
async function apiFetch(url, options) {
    const res = await fetch(url, options);
    let payload = null;
    const text = await res.text();
    try { payload = text ? JSON.parse(text) : null; } catch (e) { payload = null; }
    if (!res.ok) {
        const detail = payload && payload.detail ? payload.detail : (text || `HTTP ${res.status}`);
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return payload;
}

function postJson(url, payload) {
    return apiFetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
}

const api = {
    async listFiles(path, rootType = 'files') {
        const p = path ? `?path=${encodeURIComponent(path)}&root_type=${rootType}` : `?root_type=${rootType}`;
        return apiFetch(`/api/fs/list${p}`);
    },
    getConfig() { return apiFetch('/api/config'); },
    // folder objects carry `folder_id` (not `id`)
    getFolders() { return apiFetch('/api/vmsam/folders_list'); },
    findFolders(text) { return apiFetch(`/api/vmsam/folders_like?destination_like=${encodeURIComponent(text)}`); },
    // POST /folder/ creates OR updates (same destination_path) with the FULL
    // data; destination_path relative to CREATE_ROOT, the relay prefixes it.
    // Answers the folder object + message "Folder created" / "Folder updated".
    saveFolder(payload) { return postJson('/api/vmsam/folders', payload); },
    // POST /regex/ creates a rule, or updates an existing pattern (folder via
    // destination_path, rename_pattern, weight); example_filename is checked
    // either way (it must match and yield an episode >= 1)
    saveRegex(payload) { return postJson('/api/vmsam/regex', payload); },
    // { regex_pattern, new_regex_pattern, example_filename }: the pattern only
    replaceRegex(payload) { return postJson('/api/vmsam/regex_replace', payload); },
    getRegexes(folderId) { return apiFetch(`/api/vmsam/regex_list?folder_id=${folderId}`); },
    indexFolder(folderId) { return postJson('/api/vmsam/index_folder', { folder_id: folderId }); },
    createIncrementaller(payload) { return postJson('/api/vmsam/incrementaller', payload); },
    getIncrementallers() { return apiFetch('/api/vmsam/incrementaller_list'); },
    getFolderEpisodes(folderId) { return apiFetch(`/api/vmsam/episodes_folder?folder_id=${folderId}`); },
    // newPath is relative to CREATE_ROOT: the relay prefixes the root
    moveFolder(folderId, newPath) { return postJson('/api/vmsam/folders_move', { folder_id: folderId, new_destination_path: newPath }); },
    getFolder(folderId) { return apiFetch(`/api/vmsam/folder?folder_id=${folderId}`); },
    // renames: [{ episode_number, new_file_name }], all or nothing
    renameEpisodes(folderId, renames) { return postJson('/api/vmsam/episodes_rename', { folder_id: folderId, renames }); },
    deleteRegex(regexPattern) { return postJson('/api/vmsam/regex_delete', { regex_pattern: regexPattern }); },
    getErrors() { return apiFetch('/api/vmsam/errors'); },
    deleteError(errorFilePath) { return postJson('/api/vmsam/errors_delete', { error_file_path: errorFilePath }); },
    queueFusion(errorFilePath) { return postJson('/api/vmsam/fusion', { error_file_path: errorFilePath }); },
    getFusionStatus() { return apiFetch('/api/vmsam/fusion_status'); }
};

// --- 3. Filename analysis engine ---
//
// A file name is cut into tokens: a run of digits, a run of letters, or one
// other character. A group of names is aligned on the first one (the
// reference): tokens every name shares stay literal, a digit run that only
// changes value becomes \d+ (two digit runs pair up whatever their values, so
// an episode number stays aligned next to a changing title), anything else
// that differs becomes .+ (.* when some name has nothing there). One digit run
// is then chosen as the episode number: the user's click, else the E digits
// of a season/episode marker every name carries (SxxE, 3x01, Saison 1
// Épisode 01), else the best score. The proposed
// rename drops the wild spans, which belong to the reference file only.

const EPISODE_PLACEHOLDER = '{<episode>}';
const RESOLUTION_VALUES = new Set(['480', '576', '720', '1080', '1440', '2160', '4320']);
const CODEC_VALUES = new Set(['264', '265']);

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\\/-]/g, '\\$&');
}

function tokenizeName(name) {
    const re = /\d+|[A-Za-zÀ-ɏ]+|[\s\S]/g;
    const tokens = [];
    let m;
    while ((m = re.exec(name)) !== null) {
        tokens.push({ v: m[0], d: /^\d+$/.test(m[0]), start: m.index });
    }
    return tokens;
}

function sameToken(a, b) {
    return a.v === b.v && a.d === b.d;
}

// Alignment weight of two tokens: identical tokens weigh far more than two
// digit runs of different values, so a number only pairs with another number
// when that costs no identical match (episode numbers stay aligned even when
// the title next to them changes, and 1080 still pairs with 1080).
const MATCH_WEIGHT_SAME = 100;
// Two letter runs that differ only by case (`Of` / `of`) still pair, a little
// below identical ones: the regex then carries `[oO]` for the letters that change.
const MATCH_WEIGHT_CASE = 90;
const MATCH_WEIGHT_DIGITS = 25;
// Each run of unmatched tokens costs a little, far less than any match: among
// alignments of equal weight, the one with the fewest holes wins, so the dots
// around a changing title stay with the words they separate.
const GAP_OPEN_COST = 1;

// Same letters, same length, at least one letter in another case.
function sameButCase(a, b) {
    return !a.d && !b.d && a.v !== b.v && a.v.length === b.v.length
        && /^[A-Za-zÀ-ɏ]+$/.test(a.v) && a.v.toLowerCase() === b.v.toLowerCase();
}

function matchWeight(a, b) {
    if (sameToken(a, b)) return MATCH_WEIGHT_SAME;
    if (sameButCase(a, b)) return MATCH_WEIGHT_CASE;
    if (a.d && b.d) return MATCH_WEIGHT_DIGITS;
    return 0;
}

// Weighted longest common subsequence of two token arrays, as matched index
// pairs. A pair may join two digit runs of different values. Two tables:
// afterMatch[i][j] is the best score of a[i..], b[j..] right after a match (a
// skip opens a hole), inGap[i][j] the same inside a hole (a skip is free).
function lcsPairs(a, b) {
    const n = a.length, m = b.length;
    const afterMatch = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
    const inGap = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
    for (let i = n; i >= 0; i--) {
        for (let j = m; j >= 0; j--) {
            if (i === n || j === m) {
                afterMatch[i][j] = (i === n && j === m) ? 0 : -GAP_OPEN_COST;
                inGap[i][j] = 0;
                continue;
            }
            const w = matchWeight(a[i], b[j]);
            const take = w > 0 ? w + afterMatch[i + 1][j + 1] : -Infinity;
            const skip = Math.max(inGap[i + 1][j], inGap[i][j + 1]);
            afterMatch[i][j] = Math.max(take, skip - GAP_OPEN_COST);
            inGap[i][j] = Math.max(take, skip);
        }
    }
    const pairs = [];
    let i = 0, j = 0, gap = false;
    while (i < n && j < m) {
        const w = matchWeight(a[i], b[j]);
        const here = gap ? inGap[i][j] : afterMatch[i][j];
        // On a tie, a separator is left for later: the hole then sits before
        // the dot, and the dot stays with the word that follows the hole.
        const skipValue = Math.max(inGap[i + 1][j], inGap[i][j + 1]) - (gap ? 0 : GAP_OPEN_COST);
        const separatorTie = !/[0-9A-Za-zÀ-ɏ]/.test(a[i].v) && skipValue === here;
        if (w > 0 && here === w + afterMatch[i + 1][j + 1] && !separatorTie) { pairs.push([i, j]); i++; j++; gap = false; continue; }
        gap = true;
        if (inGap[i + 1][j] >= inGap[i][j + 1]) i++;
        else j++;
    }
    return pairs;
}

// The first season/episode marker of a token list, or null:
//   sxe:    S<season>E<episode>. `S` must be its own letter run (`.S01E03.`,
//           `s02e05`, ` S1E3 `), or the capital end of a camel-cased word
//           (`TitleS01E03`, glued); `E` must stand alone.
//   nx:     <season>x<episode>, 1-2 digits then 2-3 digits, not glued to a
//           word (`3x01`, `00x12`; never `1920x1080` nor `x264`).
//   saison: Saison|Season <n> Épisode|Episode <n>, separators in between.
// `from` is the token where the marker starts, `episode` the episode digits.
function findMarkerSpan(tokens) {
    const isWord = t => !!t && /^[A-Za-zÀ-ɏ]/.test(t.v);
    const isSep = t => !!t && !t.d && !isWord(t);
    const skipSeps = j => { while (isSep(tokens[j])) j++; return j; };
    for (let k = 0; k < tokens.length; k++) {
        const t = tokens[k];
        if (k + 3 < tokens.length) {
            const isMarkerS = /^s$/i.test(t.v) || /[a-zà-ÿ]S$/.test(t.v);
            if (isMarkerS && tokens[k + 1].d && /^e$/i.test(tokens[k + 2].v) && tokens[k + 3].d) {
                return { kind: 'sxe', from: k, glued: t.v.length > 1, episode: k + 3 };
            }
        }
        if (t.d && t.v.length <= 2 && k + 2 < tokens.length && /^x$/i.test(tokens[k + 1].v)
            && tokens[k + 2].d && /^\d{2,3}$/.test(tokens[k + 2].v) && !CODEC_VALUES.has(tokens[k + 2].v)
            && !isWord(tokens[k - 1]) && !(isWord(tokens[k + 3]) && !/^v$/i.test(tokens[k + 3].v))) {
            return { kind: 'nx', from: k, glued: false, episode: k + 2 };
        }
        if (/^(saison|season)$/i.test(t.v)) {
            let j = skipSeps(k + 1);
            if (!tokens[j] || !tokens[j].d) continue;
            j = skipSeps(j + 1);
            if (!tokens[j] || !/^[ée]pisode$/i.test(tokens[j].v)) continue;
            j = skipSeps(j + 1);
            if (tokens[j] && tokens[j].d) return { kind: 'saison', from: k, glued: false, episode: j };
        }
    }
    return null;
}

// Index of the episode digit run of a season/episode marker, or -1.
function findEpisodeMarker(tokens) {
    const marker = findMarkerSpan(tokens);
    return marker ? marker.episode : -1;
}

// A short literal run (at most three tokens) squeezed between two wild spans
// is a coincidence of the titles (`.Is.` in two different titles): make it
// wild too, so the regex holds one `.+` and the rename drops it. A changing
// digit run glued after a wild word is the end of that word (`ABCD1234`, a
// CRC): wild too, unless that word is an episode prefix (`E`, `Ep`). The
// season/episode marker is never absorbed.
function absorbBetweenWilds(slots, protectedFrom, protectedTo) {
    const isProtected = k => k >= protectedFrom && k <= protectedTo;
    for (let i = 1; i < slots.length; i++) {
        const left = slots[i - 1];
        const wildWord = left.wild && /^[A-Za-zÀ-ɏ]{2,}$/.test(left.v) && !/(^|[^a-z])(ep|episode)$/i.test(left.v);
        if (slots[i].d && slots[i].varying && !slots[i].wild && wildWord && !isProtected(i)) {
            slots[i].wild = true;
        }
    }
    let lastWild = -1;
    for (let i = 0; i < slots.length; i++) {
        if (!slots[i].wild) continue;
        const gap = i - lastWild - 1;
        if (lastWild >= 0 && gap > 0 && gap <= 3) {
            let ok = true;
            for (let k = lastWild + 1; k < i; k++) {
                if (isProtected(k)) ok = false;
                if (slots[k].d && !slots[k].varying) ok = false;
            }
            if (ok) for (let k = lastWild + 1; k < i; k++) { slots[k].wild = true; slots[k].varying = true; }
        }
        lastWild = i;
    }
}

// Template = reference tokens annotated with what the other names taught us.
//   varying: the token differs in at least one other name
//   wild:    the difference is not "one digit run for another" -> becomes .+
//   optional: wild, and absent from at least one other name -> .*
//   extraBefore: another name carries tokens here that the reference lacks
//                (empty in the reference, so always .*)
//   caseFold: letters of a literal word written in another case in some
//             name (`Of` / `of`): one flag per letter, emitted as `[oO]`
//   markerIndex: the episode digits of a season/episode marker (SxxE, NxNN,
//                Saison n Épisode n: findMarkerSpan) every name carries,
//                aligned in every name (null otherwise)
function buildTemplate(names) {
    const ref = tokenizeName(names[0]);
    const slots = ref.map(t => ({ v: t.v, d: t.d, start: t.start, end: t.start + t.v.length, varying: false, wild: false, optional: false, extraBefore: false }));
    let extraAfterEnd = false;
    const refSpan = findMarkerSpan(ref);
    const refMarker = refSpan ? refSpan.episode : -1;
    let markerAligned = refMarker >= 0;
    const alignments = [];

    for (const name of names.slice(1)) {
        const other = tokenizeName(name);
        const pairs = lcsPairs(ref, other);
        alignments.push({ other, pairs });
        if (markerAligned) {
            const otherSpan = findMarkerSpan(other);
            const otherMarker = otherSpan && otherSpan.kind === refSpan.kind ? otherSpan.episode : -1;
            markerAligned = otherMarker >= 0 && pairs.some(p => p[0] === refMarker && p[1] === otherMarker);
        }
        let prev = [-1, -1];
        for (const pair of [...pairs, [ref.length, other.length]]) {
            const [i2, j2] = pair;
            const [i1, j1] = prev;
            if (i2 < ref.length && j2 < other.length && sameButCase(ref[i2], other[j2])) {
                // Only the case changes: remember which letters, the token stays literal.
                const fold = slots[i2].caseFold || (slots[i2].caseFold = new Array(ref[i2].v.length).fill(false));
                for (let k = 0; k < fold.length; k++) if (ref[i2].v[k] !== other[j2].v[k]) fold[k] = true;
            } else if (i2 < ref.length && j2 < other.length && !sameToken(ref[i2], other[j2])) slots[i2].varying = true;
            const refGap = i2 - i1 - 1;
            const otherGap = j2 - j1 - 1;
            if (refGap > 0 || otherGap > 0) {
                const otherTokens = other.slice(j1 + 1, j2);
                const otherAllDigits = otherTokens.length > 0 && otherTokens.every(t => t.d);
                if (refGap === 0) {
                    if (i2 < slots.length) slots[i2].extraBefore = true;
                    else extraAfterEnd = true;
                } else {
                    for (let k = i1 + 1; k < i2; k++) {
                        slots[k].varying = true;
                        if (!(refGap === 1 && otherGap === 1 && slots[k].d && otherAllDigits)) slots[k].wild = true;
                        // Nothing in the other name here: the span may be empty.
                        if (otherGap === 0) slots[k].optional = true;
                    }
                }
            }
            prev = pair;
        }
    }
    const markerIndex = markerAligned && !slots[refMarker].wild ? refMarker : null;
    absorbBetweenWilds(slots, markerIndex === null ? -1 : refSpan.from, markerIndex === null ? -1 : markerIndex);
    settleWildRuns(slots, alignments);
    return { name: names[0], slots, extraAfterEnd, count: names.length, markerIndex };
}

// Each maximal run of wild slots, once absorbed, is measured again against
// every name: the tokens a name holds between the two literal tokens around
// the run. A separator that opens (or closes) that text in every name leaves
// the run and stays literal (`S01E03 - Title` keeps ` - ` whatever the
// titles), and the run is optional (.*) only when that text is empty in some
// name. A run whose neighbours are not aligned in some name is left as is.
function settleWildRuns(slots, alignments) {
    const isSeparator = v => !/[0-9A-Za-zÀ-ɏ]/.test(v);
    const maps = alignments.map(({ other, pairs }) => ({ other, at: new Map(pairs) }));
    let a = 0;
    while (a < slots.length) {
        if (!slots[a].wild) { a++; continue; }
        let b = a;
        while (b + 1 < slots.length && slots[b + 1].wild) b++;
        const next = b + 1;
        const spans = maps.map(({ other, at }) => {
            const left = a === 0 ? -1 : at.get(a - 1);
            const right = next === slots.length ? other.length : at.get(next);
            return left === undefined || right === undefined ? null : { other, from: left + 1, to: right };
        });
        if (spans.every(Boolean)) {
            while (a < b && isSeparator(slots[a].v) && spans.every(s => s.from < s.to && s.other[s.from].v === slots[a].v)) {
                Object.assign(slots[a], { wild: false, varying: false, optional: false, extraBefore: false });
                spans.forEach(s => { s.from++; });
                a++;
            }
            while (b > a && isSeparator(slots[b].v) && spans.every(s => s.from < s.to && s.other[s.to - 1].v === slots[b].v)) {
                Object.assign(slots[b], { wild: false, varying: false, optional: false, extraBefore: false });
                spans.forEach(s => { s.to--; });
                b--;
            }
            const optional = spans.some(s => s.from >= s.to);
            for (let k = a; k <= b; k++) slots[k].optional = optional;
            // The text of every name is counted in the run: no extra before what follows.
            if (next < slots.length) slots[next].extraBefore = false;
        }
        a = next;
    }
}

// Digit runs that can carry the episode number: every non-wild digit slot.
function numberSlots(template) {
    return template.slots.map((s, i) => ({ s, i })).filter(x => x.s.d && !x.s.wild).map(x => x.i);
}

// Why a digit run is (not) the episode number. Positive is good.
function scoreEpisodeSlot(template, index) {
    const slot = template.slots[index];
    const before = template.name.slice(0, slot.start);
    const after = template.name.slice(slot.end);
    const value = slot.v;
    let score = 0;
    const reasons = [];
    const add = (points, why) => { score += points; reasons.push(`${points > 0 ? '+' : ''}${points} ${why}`); };

    const extensionStart = template.name.lastIndexOf('.');
    if (extensionStart > 0 && slot.start > extensionStart && !/[\s._\-\[\]()]/.test(after)) add(-200, 'inside the file extension');

    // A season/episode marker in every name of the group wins outright.
    if (template.markerIndex === index) add(1000, 'season/episode marker in every name');
    else if (/S\d{1,3}[ ._-]?E$/i.test(before)) add(100, 'SxxEyy');
    else if (/(^|[^0-9A-Za-zÀ-ɏ])\d{1,2}x$/i.test(before) && /^\d{2,3}$/.test(value)) add(100, 'NxNN');
    else if (/(^|[\s._\-\[(])(ep|episode|épisode|e)[\s._\-]*$/i.test(before)) add(50, 'episode word');
    else if (/[\s._]-[\s._]$/.test(before)) add(30, 'after dash');
    else if (/[\s._\-\])]$/.test(before)) add(10, 'after separator');
    else if (/[A-Za-z]$/.test(before)) add(-5, 'glued to a word');

    if (/^(p|i)([\s._\-\]\)]|$)/i.test(after) || /^(bit|fps|kbps|hz|khz)/i.test(after)) add(-100, 'resolution/rate unit');
    if (/(x|h\.?|hevc|av)$/i.test(before) && CODEC_VALUES.has(value)) add(-100, 'codec');
    if (RESOLUTION_VALUES.has(value)) add(-60, 'resolution value');
    if (CODEC_VALUES.has(value)) add(-60, 'codec value');
    if (/v$/i.test(before)) add(-80, 'version tag');
    if (/(^|[\s._\-\[(])S$/i.test(before)) add(-60, 'season number');
    if (/^(19|20)\d\d$/.test(value)) add(-80, 'year');
    if (value.length > 4) add(-40, 'too long');
    if (value.length === 8) add(-30, 'looks like a CRC');
    if (/^0\d$/.test(value)) add(15, 'zero padded');
    else if (value.length <= 3) add(5, 'short');
    const opens = (before.match(/[\[(]/g) || []).length;
    const closes = (before.match(/[\])]/g) || []).length;
    if (opens > closes) add(-20, 'inside brackets');

    if (template.count > 1) {
        if (slot.varying) add(40, 'changes between files');
        else add(-25, 'same in every file');
    }
    return { score, reasons };
}

function pickEpisodeSlot(template) {
    const candidates = numberSlots(template).map(i => ({ index: i, value: template.slots[i].v, ...scoreEpisodeSlot(template, i) }));
    candidates.sort((a, b) => b.score - a.score || b.index - a.index);
    // Below -100 every digit run was ruled out (extension, resolution, codec):
    // better no guess than a wrong one, the user picks a chip.
    const best = candidates.length && candidates[0].score > -100 ? candidates[0].index : null;
    return { best, candidates };
}

function buildRegex(template, episodeIndex) {
    let out = '^';
    let pendingWild = false;
    let pendingOptional = false;
    // A wild span right before the episode is lazy, or it would eat the
    // leading digits of the episode number.
    const flush = (beforeEpisode) => {
        if (pendingWild) out += (pendingOptional ? '.*' : '.+') + (beforeEpisode ? '?' : '');
        pendingWild = false;
        pendingOptional = false;
    };
    template.slots.forEach((slot, i) => {
        if (slot.extraBefore) { pendingWild = true; pendingOptional = true; }
        if (slot.wild) { pendingWild = true; if (slot.optional) pendingOptional = true; return; }
        flush(i === episodeIndex);
        if (i === episodeIndex) out += '(?P<episode>\\d+)';
        else if (slot.d && slot.varying) out += '\\d+';
        else if (slot.caseFold) out += [...slot.v].map((ch, k) => (slot.caseFold[k] ? `[${ch.toLowerCase()}${ch.toUpperCase()}]` : escapeRegex(ch))).join('');
        else out += escapeRegex(slot.v);
    });
    if (template.extraAfterEnd) { pendingWild = true; pendingOptional = true; }
    flush();
    return out + '$';
}

// The reference name with the episode digits replaced by the placeholder.
// Wild spans (a per-episode title, a changing tag) belong to the reference
// file only: they are dropped, and the separators left on both sides of the
// hole collapse into one (the left one, or the dot of the extension).
// Then the episode stands out (owner's rule): the episode with the marker
// glued to it (`S01E`, `1x`, `E`, `Ep`) is the unit, and a separator run
// on either side of it that holds no dash becomes ` - `
// (`Show.S01E03.1080p.mkv` -> `Show - S01E{<episode>} - 1080p.mkv`). A run
// that already holds a dash is kept (no doubled ` - `), and nothing is added
// at the start, at the end, before the extension or where the unit touches a
// bracket or a word directly.
function buildRename(template, episodeIndex) {
    if (episodeIndex === null || episodeIndex === undefined) return template.name;
    const GAP = '\u0000';
    const UNIT_START = '\u0001';
    const UNIT_END = '\u0002';
    const slots = template.slots;
    const isGlued = slot => !slot.wild && /^[0-9A-Za-zÀ-ɏ]+$/.test(slot.v);
    const isPlainSeparator = slot => !slot.wild && /^[\s._]$/.test(slot.v);
    let unitStart = episodeIndex;
    while (unitStart > 0 && isGlued(slots[unitStart - 1])) unitStart--;
    // A separated episode word belongs to the unit too (`Ep 03`, `Episode.03`),
    // and so does a whole `Saison 1 Épisode 03` marker.
    let word = unitStart;
    while (word > 0 && isPlainSeparator(slots[word - 1])) word--;
    if (word < unitStart && word > 0 && !slots[word - 1].wild && /^(e|ep|episode|épisode)$/i.test(slots[word - 1].v)) unitStart = word - 1;
    const marker = findMarkerSpan(slots);
    if (marker && marker.kind === 'saison' && marker.episode === episodeIndex && !slots.slice(marker.from, episodeIndex).some(s => s.wild)) unitStart = Math.min(unitStart, marker.from);
    let out = '';
    template.slots.forEach((slot, i) => {
        if (slot.wild) { if (!out.endsWith(GAP)) out += GAP; return; }
        if (i === unitStart) out += UNIT_START;
        out += i === episodeIndex ? EPISODE_PLACEHOLDER + UNIT_END : slot.v;
    });
    if (out.includes(GAP)) {
        // A bracket pair that only held a dropped span goes with it.
        let previous;
        do {
            previous = out;
            out = out.replace(/[\[(][\s._\-]*\u0000[\s._\-]*[\])]/g, GAP).replace(/\u0000+/g, GAP);
        } while (out !== previous);
        out = out.replace(/([\s._\-]*)\u0000([\s._\-]*)/g, (match, left, right, offset, whole) => {
            const before = whole.slice(0, offset).replace(/[\u0001\u0002]/g, '');
            const after = whole.slice(offset + match.length).replace(/[\u0001\u0002]/g, '');
            if (!before || !after) return '';
            if (right.endsWith('.') && /^[A-Za-z0-9]{1,5}$/.test(after)) return '.';
            return left || right;
        });
    }
    const DASH = ' - ';
    out = out.replace(/([\s._\-]*)\u0001/, (match, run, offset) => {
        if (!run || offset === 0 || run.includes('-')) return run;
        return DASH;
    });
    out = out.replace(/\u0002([\s._\-]*)/, (match, run, offset, whole) => {
        const after = whole.slice(offset + match.length);
        if (!run || !after || run.includes('-')) return run;
        if (run.endsWith('.') && /^[A-Za-z0-9]{1,5}$/.test(after)) return run;
        return DASH;
    });
    return out.replace(/[\u0001\u0002]/g, '');
}

// Python named groups -> JS named groups, so the browser can test the pattern.
function toJsRegex(pythonPattern) {
    return new RegExp(pythonPattern.replace(/\(\?P</g, '(?<'));
}

function extractEpisode(pythonPattern, name) {
    try {
        const match = name.match(toJsRegex(pythonPattern));
        if (match && match.groups && match.groups.episode !== undefined) return match.groups.episode;
        if (match && match.length > 1 && match[1] !== undefined) return match[1];
        return null;
    } catch (e) {
        return null;
    }
}

function testPattern(pythonPattern, names) {
    let syntaxError = null;
    try { toJsRegex(pythonPattern); } catch (e) { syntaxError = e.message; }
    const perFile = names.map(name => {
        const episode = syntaxError ? null : extractEpisode(pythonPattern, name);
        const valid = episode !== null && /^\d+$/.test(episode) && parseInt(episode, 10) > 0;
        return { name, episode, valid };
    });
    return { syntaxError, perFile, allValid: !syntaxError && perFile.every(f => f.valid) };
}

// A wild span right after the episode number (`(?P<episode>\d+).+`) starts,
// when every name agrees, with the separator that follows the number in the
// reference (` - `, `.`, `_`, `[`...): keep that separator literal and push the
// wild span after it, `S01E(?P<episode>\d+)\..+`. An optional span (`.*`)
// whose text ends on that same separator becomes `\.(?:.+\.)?`. A variant is
// kept only when it extracts, from every name, the same episode as before.
function anchorAfterEpisode(regex, template, episodeIndex, names) {
    if (episodeIndex === null || episodeIndex === undefined) return regex;
    const group = '(?P<episode>\\d+)';
    const at = regex.indexOf(group);
    if (at < 0) return regex;
    const head = regex.slice(0, at + group.length);
    const wild = regex.slice(head.length, head.length + 2);
    if (wild !== '.+' && wild !== '.*') return regex;
    const rest = regex.slice(head.length + 2);
    if (rest.startsWith('?')) return regex;
    const run = (template.name.slice(template.slots[episodeIndex].end).match(/^[^0-9A-Za-zÀ-ɏ]{1,3}/) || [''])[0];
    const expected = names.map(n => extractEpisode(regex, n));
    const same = candidate => {
        try { toJsRegex(candidate); } catch (e) { return false; }
        return names.every((n, k) => extractEpisode(candidate, n) === expected[k]);
    };
    for (let length = run.length; length > 0; length--) {
        const literal = escapeRegex(run.slice(0, length));
        const candidates = [head + literal + '.+' + rest, head + literal + '.*' + rest];
        if (rest.startsWith(literal)) candidates.push(head + literal + '(?:.+' + literal + ')?' + rest.slice(literal.length));
        const found = candidates.find(same);
        if (found) return found;
    }
    return regex;
}

// The whole proposal for one group.
function analyzeGroup(names, forcedEpisodeIndex) {
    const template = buildTemplate(names);
    const pick = pickEpisodeSlot(template);
    let episodeIndex = pick.best;
    if (forcedEpisodeIndex !== null && forcedEpisodeIndex !== undefined && numberSlots(template).includes(forcedEpisodeIndex)) {
        episodeIndex = forcedEpisodeIndex;
    }
    const regex = anchorAfterEpisode(buildRegex(template, episodeIndex), template, episodeIndex, names);
    return {
        template,
        episodeIndex,
        candidates: pick.candidates,
        regex,
        rename: buildRename(template, episodeIndex),
        check: testPattern(regex, names)
    };
}

// Default grouping. A word is a run between separators, compared
// case-insensitively; the extension is not a word but must be equal.
//  1. Files first gather by their prefix: the words before a season/episode
//     marker (findMarkerSpan: SxxE, 3x01, Saison 1 Épisode 01) and the marker
//     kind, or the first two words without a marker; plus the extension.
//  2. Inside a prefix, the tail (last three words after the marker, or of the
//     whole name; CRC words `[0-9a-f]{8}` left out) splits files only when at
//     least two files share it: two releases of a show stay apart, while a
//     season whose names end with the episode title, unique per file, stays
//     together in one pooled group.
//  3. A file with no marker and no episode-like word leaves a group where
//     other files have one (a special among numbered episodes): own card.
//  4. Each group is then refined pair by pair (splitGroupByPairs): sub-groups
//     of two names or more that keep more words literal than the whole group,
//     outside the per-episode title (two releases, two seasons, OP and ED).
// An episode-like word (`01`, `E05`, `ep3`, `07v2`: at most three digits)
// only counts as "a number here" in a key. Groups keep the order of their
// first file, names inside a group keep input order.
const NAME_WORD_SEPARATORS = /[\s.\/,\-()_;:\[\]{}+~!']+/;
const NAME_EXTENSION = /\.([A-Za-z0-9]{1,5})$/;
const EPISODE_LIKE_WORD = /^(?:e|ep)?\d{1,3}(?:v\d+)?$/;
const CRC_WORD = /^[0-9a-f]{8}$/;

function nameWords(text) {
    return text.split(NAME_WORD_SEPARATORS).filter(Boolean).map(w => w.toLowerCase());
}

// Separator runs of a text, in order: `[Grp] Show - ` -> ['[', '] ', ' - '].
// Part of the prefix key, so `Show D 01` and `Show_D_01` never group: the
// same words written with another separator style are another release, and
// one regex cannot cover both.
const NAME_SEPARATOR_RUNS = /[\s.\/,\-()_;:\[\]{}+~!']+/g;
function separatorRuns(text) {
    return text.match(NAME_SEPARATOR_RUNS) || [];
}

// Separator runs around the first `count` words: before, between and right after them.
function leadingSeparatorRuns(text, count) {
    const runs = [];
    let words = 0;
    for (const part of text.split(/([\s.\/,\-()_;:\[\]{}+~!']+)/)) {
        if (part === '') continue;
        if (NAME_WORD_SEPARATORS.test(part) && !/[^\s.\/,\-()_;:\[\]{}+~!']/.test(part)) {
            runs.push(part);
            if (words >= count) break;
        } else {
            words++;
            if (words > count) break;
        }
    }
    return runs;
}

// { prefix, tail, numbered } of one name; prefix and tail are string keys.
function nameShape(name) {
    const ext = name.match(NAME_EXTENSION);
    const extension = ext ? ext[1].toLowerCase() : '';
    const stem = ext ? name.slice(0, ext.index) : name;
    const tokens = tokenizeName(stem);
    const marker = findMarkerSpan(tokens);
    const tailOf = words => JSON.stringify(words.filter(w => !CRC_WORD.test(w)).slice(-3));
    if (marker) {
        const first = tokens[marker.from];
        const markerStart = marker.glued ? first.start + first.v.length - 1 : first.start; // the S of `TitleS01E03`
        const markerEnd = tokens[marker.episode].start + tokens[marker.episode].v.length;
        // Whatever is glued after the episode digits belongs to the marker word.
        const after = stem.slice(markerEnd).replace(/^[^\s.\/,\-()_;:\[\]{}+~!']+/, '');
        const before = stem.slice(0, markerStart);
        const separators = [...separatorRuns(before), (after.match(/^[\s.\/,\-()_;:\[\]{}+~!']+/) || [''])[0]];
        return {
            prefix: JSON.stringify([marker.kind, nameWords(before), separators, extension]),
            tail: tailOf(nameWords(after)),
            numbered: true
        };
    }
    const raw = nameWords(stem);
    const words = raw.map(w => (EPISODE_LIKE_WORD.test(w) ? '#' : w));
    return {
        prefix: JSON.stringify(['plain', words.slice(0, 2), leadingSeparatorRuns(stem, 2), extension]),
        tail: tailOf(words),
        numbered: raw.some(w => EPISODE_LIKE_WORD.test(w))
    };
}

// Default grouping: the first pass by name shape, then the pairwise refinement.
function groupByNamePattern(names) {
    const order = new Map();
    names.forEach((n, i) => { if (!order.has(n)) order.set(n, i); });
    return refineGroupsByPairs(groupByNameShape(names)).sort((x, y) => order.get(x[0]) - order.get(y[0]));
}

// First pass of groupByNamePattern: prefix, tail, numbered (see above).
function groupByNameShape(names) {
    const order = new Map();
    names.forEach((n, i) => { if (!order.has(n)) order.set(n, i); });
    const shapes = new Map(names.map(n => [n, nameShape(n)]));
    const byPrefix = new Map();
    for (const name of names) {
        const prefix = shapes.get(name).prefix;
        if (!byPrefix.has(prefix)) byPrefix.set(prefix, []);
        byPrefix.get(prefix).push(name);
    }
    const result = [];
    for (const members of byPrefix.values()) {
        const tailCount = new Map();
        members.forEach(n => { const t = shapes.get(n).tail; tailCount.set(t, (tailCount.get(t) || 0) + 1); });
        const sub = new Map();
        for (const name of members) {
            const tail = shapes.get(name).tail;
            const key = tailCount.get(tail) >= 2 ? 'tail' + tail : 'pool';
            if (!sub.has(key)) sub.set(key, []);
            sub.get(key).push(name);
        }
        for (const group of sub.values()) {
            const numbered = group.filter(n => shapes.get(n).numbered);
            if (numbered.length && numbered.length < group.length) {
                result.push(numbered);
                group.filter(n => !shapes.get(n).numbered).forEach(n => result.push([n]));
            } else {
                result.push(group);
            }
        }
    }
    return result.sort((x, y) => order.get(x[0]) - order.get(y[0]));
}

// Second pass: inside one group of the first pass, look for sub-groups whose
// names share more than the whole group does (two releases of a show the
// first pass pooled). What a set of names "shares" is the multiset of words
// its template keeps literal (letter runs of two letters or more, fixed digit
// runs, lowercased); a set "gains" the words it keeps beyond those of the
// whole group. Every pair of names is aligned; pairs that gain the same words
// form a class; classes become sub-groups, largest first, each file in the
// first class that takes it. A file no class took joins the sub-group that
// still gains with it. The split is kept only when every sub-group has at
// least two names and gains a word, and it leaves no more groups without a
// valid episode than before. A season whose names differ by the per-episode
// title only gains nothing outside the titles (short runs between two wild
// spans are absorbed) and stays whole.
const REFINE_MIN_GROUP = 4;
// Every pair is aligned: about 0.5 s for 150 names, so larger groups are left as they are.
const REFINE_MAX_GROUP = 200;

// With `anchoredOnly`, words of a literal run that a wild span closes on the
// right are left out: they open a per-episode title (`- The .+`, `(Il faut .+)`),
// and titles sharing their first word are no release of their own. A run
// ends at a wild span or a changing digit run.
function literalWords(template, anchoredOnly) {
    const words = new Map();
    const slots = template.slots;
    let k = 0;
    while (k < slots.length) {
        if (slots[k].wild || (slots[k].d && slots[k].varying)) { k++; continue; }
        let end = k;
        while (end < slots.length && !slots[end].wild && !(slots[end].d && slots[end].varying)) end++;
        const titleStart = anchoredOnly && end < slots.length && slots[end].wild;
        for (let i = k; i < end && !titleStart; i++) {
            const slot = slots[i];
            if (!slot.d && !/^[A-Za-zÀ-ɏ]{2,}$/.test(slot.v)) continue;
            const word = slot.v.toLowerCase();
            words.set(word, (words.get(word) || 0) + 1);
        }
        k = end;
    }
    return words;
}

// Words of `words` beyond those of `base` (multiset difference), sorted.
function gainedWords(words, base) {
    const gained = [];
    for (const [word, count] of words) for (let k = base.get(word) || 0; k < count; k++) gained.push(word);
    return gained.sort();
}

function splitGroupByPairs(names) {
    if (names.length < REFINE_MIN_GROUP || names.length > REFINE_MAX_GROUP) return [names];
    const baseWords = literalWords(buildTemplate(names));
    const gainOf = indexes => gainedWords(literalWords(buildTemplate([...indexes].sort((a, b) => a - b).map(k => names[k])), true), baseWords);
    const classes = new Map();
    for (let i = 0; i < names.length; i++) {
        for (let j = i + 1; j < names.length; j++) {
            const gained = gainOf([i, j]);
            if (!gained.length) continue;
            const key = JSON.stringify(gained);
            if (!classes.has(key)) classes.set(key, { gained, members: new Set(), first: i });
            classes.get(key).members.add(i).add(j);
        }
    }
    if (!classes.size) return [names];
    const ordered = [...classes.values()].sort((a, b) => b.members.size - a.members.size || b.gained.length - a.gained.length || a.first - b.first);
    const taken = new Array(names.length).fill(false);
    const parts = [];
    for (const c of ordered) {
        const free = [...c.members].filter(k => !taken[k]).sort((a, b) => a - b);
        if (free.length < 2) continue;
        free.forEach(k => { taken[k] = true; });
        parts.push(free);
    }
    const unplaced = [];
    names.forEach((_, k) => {
        if (taken[k]) return;
        let best = -1, bestGain = 0;
        parts.forEach((part, p) => {
            const gain = gainOf([...part, k]).length;
            if (gain > bestGain) { best = p; bestGain = gain; }
        });
        if (best >= 0) parts[best].push(k);
        else unplaced.push(k);
    });
    if (unplaced.length) parts.push(unplaced);
    if (parts.length < 2 || parts.some(part => part.length < 2)) return [names];
    const split = parts.map(part => part.sort((a, b) => a - b).map(k => names[k]));
    if (split.some(part => !partGain(names, part))) return [names];
    const proposals = split.map(part => analyzeGroup(part));
    if (new Set(proposals.map(p => p.regex)).size < proposals.length) return [names];
    if (proposals.filter(p => !p.check.allValid).length > (analyzeGroup(names).check.allValid ? 0 : 1)) return [names];
    return split;
}

// How many words a sub-group keeps literal that the whole group does not,
// both aligned on the sub-group's first name so their slots match. The title
// area is the wild span of the whole group right after the episode number
// (separators and at most one short shared word in between): a word kept at its start opens a per-episode title
// (`- La danse .+`, an arc name) and does not count, while a word kept after a
// part that still changes inside the sub-group (`- .+ [GrpA]`) does. Neither
// does a fixed number after a wild span (`Title – Chapitre 2`): a season or a
// resolution comes before any title.
function partGain(groupNames, part) {
    const sub = buildTemplate(part).slots;
    const whole = buildTemplate([part[0], ...groupNames.filter(n => n !== part[0])]);
    const slots = whole.slots;
    const episode = whole.markerIndex !== null ? whole.markerIndex : pickEpisodeSlot(whole).best;
    const loose = s => s.wild || (s.d && s.varying);
    const isSeparator = s => !/[0-9A-Za-zÀ-ɏ]/.test(s.v);
    const regionStart = k => { while (k > 0 && slots[k - 1].wild) k--; return k; };
    // Separators, and at most one short word every title shares (`La`, `The`).
    const isTitleRegion = start => {
        let j = start - 1;
        let words = 0;
        while (j >= 0 && !loose(slots[j]) && (isSeparator(slots[j]) || (!slots[j].d && slots[j].v.length <= 4 && ++words <= 1))) j--;
        return episode !== null && j === episode;
    };
    let gain = 0;
    slots.forEach((slot, k) => {
        if (k === episode || !loose(slot) || loose(sub[k])) return;
        if (!sub[k].d && !/^[A-Za-zÀ-ɏ]{2,}$/.test(sub[k].v)) return;
        if (slot.wild) {
            const start = regionStart(k);
            if (isTitleRegion(start) && !sub.slice(start, k).some(s => s.wild)) return;
        } else {
            let j = k - 1;
            while (j >= 0 && !loose(slots[j])) j--;
            if (j >= 0 && slots[j].wild) return;
        }
        gain++;
    });
    return gain;
}

function refineGroupsByPairs(groups) {
    return groups.flatMap(splitGroupByPairs);
}

// Former default grouping (same first five characters), kept for callers
// outside this file; the UI uses groupByNamePattern.
function groupByPrefix(names, length = 5) {
    const groups = new Map();
    for (const name of names) {
        const key = name.slice(0, length);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(name);
    }
    return Array.from(groups.values());
}

function padEpisode(number) {
    return String(number).padStart(2, '0');
}

// --- 4. Tabs ---

const TAB_INIT = {
    'folder-tab': () => {
        const container = document.getElementById('dir-browser');
        if (!container.children.length || container.textContent.includes('Loading')) loadDirBrowser();
    },
    'regex-tab': () => initRegexTab(),
    'index-tab': () => initIndexTab(),
    'move-tab': () => initMoveTab(),
    'incr-tab': () => initIncrementallerTab(),
    'episodes-tab': () => initEpisodesTab(),
    'settings-tab': () => initSettingsTab(),
    'rules-tab': () => initRulesTab(),
    'errors-tab': () => initErrorsTab()
};

// Two sections share the page: Shows (this file) and Movies (movie.js).
// Each tab registers its init in TAB_INIT; movie.js adds its own entries.
// The section of a tab is the data-section of the nav holding its button.
const SECTION_HOME = { shows: 'home-tab', movies: 'movie-home-tab' };
let currentSection = 'shows';

function switchSection(section) {
    currentSection = section;
    document.querySelectorAll('.section-switch button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === section);
    });
    document.querySelectorAll('.tab-nav').forEach(nav => {
        nav.classList.toggle('hidden', nav.dataset.section !== section);
    });
    const home = SECTION_HOME[section];
    if (home && document.getElementById(home)) switchTab(home);
}

function switchTab(tabId) {
    document.querySelectorAll('main > section').forEach(el => el.classList.add('hidden'));
    document.getElementById(tabId).classList.remove('hidden');
    document.querySelectorAll('.tab-nav button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    const button = document.querySelector(`.tab-nav button[data-tab="${tabId}"]`);
    const section = button ? button.closest('.tab-nav').dataset.section : null;
    if (section && section !== currentSection) {
        currentSection = section;
        document.querySelectorAll('.section-switch button').forEach(btn => btn.classList.toggle('active', btn.dataset.section === section));
        document.querySelectorAll('.tab-nav').forEach(nav => nav.classList.toggle('hidden', nav.dataset.section !== section));
    }
    if (TAB_INIT[tabId]) TAB_INIT[tabId]();
}

// The header title (a button): back to the home tab of the current section.
function goToSectionHome() {
    const home = SECTION_HOME[currentSection];
    if (home && document.getElementById(home)) switchTab(home);
}

// Home tiles of both sections: `.card.home-tile[data-tab]` with role="button"
// and tabindex="0", no onclick: the whole tile opens its tab, by a click,
// Enter or Space (style.css gives the pointer, hover and focus ring).
document.addEventListener('click', (e) => {
    const tile = e.target.closest ? e.target.closest('.home-tile[data-tab]') : null;
    if (tile) switchTab(tile.dataset.tab);
});
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const tile = e.target.closest ? e.target.closest('.home-tile[data-tab]') : null;
    if (!tile || tile !== e.target) return;
    e.preventDefault();
    switchTab(tile.dataset.tab);
});

// --- 5. Folder Creation ---

async function loadDirBrowser(path = '') {
    const container = document.getElementById('dir-browser');
    container.innerHTML = '<div class="loading-message" style="padding:1rem; text-align:center;">Loading...</div>';

    try {
        const items = await api.listFiles(path, 'create');
        container.innerHTML = '';

        if (path) {
            const parts = path.split('/');
            const parentPath = parts.slice(0, -1).join('/');
            const backBtn = el('div', 'dir-item text-blue');
            backBtn.appendChild(el('span', null, '📁 ..'));
            backBtn.onclick = () => loadDirBrowser(parentPath);
            container.appendChild(backBtn);
        }

        const dirs = items.filter(i => i.is_dir);
        if (dirs.length === 0) {
            container.appendChild(el('div', 'text-muted text-center p-2', 'No folders found'));
        }

        dirs.forEach(item => {
            const row = el('div', `dir-item ${state.currentBaseDir === item.path ? 'selected' : ''}`);
            row.onclick = (e) => selectDir(item.path, e);
            row.ondblclick = () => loadDirBrowser(item.path);

            const name = el('div', 'dir-name flex-1');
            name.appendChild(el('span', null, `📁 ${item.name}`));
            const actions = el('div', 'dir-actions');
            const selectBtn = el('button', 'btn btn-sm btn-secondary', 'Select');
            selectBtn.onclick = (e) => selectDir(item.path, e);
            const openBtn = el('button', 'btn btn-sm btn-secondary', 'Open →');
            openBtn.onclick = (e) => { e.stopPropagation(); loadDirBrowser(item.path); };
            actions.append(selectBtn, openBtn);
            row.append(name, actions);
            container.appendChild(row);
        });
    } catch (e) {
        console.error('Directory loading error:', e);
        container.innerHTML = '';
        const err = el('div', 'text-accent text-sm p-4 text-center', `Error: ${e.message}`);
        const retry = el('button', 'btn btn-sm btn-primary mt-2', 'Retry');
        retry.onclick = () => loadDirBrowser(path);
        err.appendChild(document.createElement('br'));
        err.appendChild(retry);
        container.appendChild(err);
    }
}

function selectDir(path, event) {
    event.stopPropagation();
    state.currentBaseDir = path;
    updatePreview();

    const container = document.getElementById('dir-browser');
    Array.from(container.children).forEach(child => child.classList.remove('selected'));
    const row = event.target.closest('.dir-item');
    if (row) row.classList.add('selected');
}

function updatePreview() {
    const series = document.getElementById('series-name').value.trim() || '{SeriesName}';
    const tvdb = document.getElementById('tvdb-id').value.trim() || '{TVDB_ID}';
    const sub = document.getElementById('subfolder').value.trim();
    const base = state.currentBaseDir || '[Select Base Dir]';

    const fullPath = `${base}/${series} {tvdb-${tvdb}} [tvdb-${tvdb}] [tvdbid-${tvdb}]/${sub}`;
    document.getElementById('path-preview').textContent = fullPath;
}

async function submitFolder() {
    const seriesName = document.getElementById('series-name').value.trim();
    const tvdbId = document.getElementById('tvdb-id').value.trim();
    const subfolder = document.getElementById('subfolder').value.trim();

    if (!state.currentBaseDir || !seriesName || !tvdbId) {
        showToast('Please fill all fields and select a base directory.', 'error');
        return;
    }

    const destinationPath = `${state.currentBaseDir}/${seriesName} {tvdb-${tvdbId}} [tvdb-${tvdbId}] [tvdbid-${tvdbId}]/${subfolder}`;

    const parseIntSafe = (id, def) => {
        const v = parseInt(document.getElementById(id).value, 10);
        return isNaN(v) ? def : v;
    };
    const parseFloatSafe = (id, def) => {
        const v = parseFloat(document.getElementById(id).value);
        return isNaN(v) ? def : v;
    };

    const payload = {
        destination_path: destinationPath,
        original_language: document.getElementById('orig-lang').value.trim() || "en",
        number_cut: parseIntSafe('number-cut', 10),
        cut_file_to_get_delay_second_method: parseFloatSafe('cut-delay', 2.0),
        max_episode_number: parseIntSafe('max-ep', 12)
    };

    try {
        const result = await api.saveFolder(payload);
        showToast(result && result.message ? result.message : 'Folder created successfully!', 'success');
        folderCache = null; // the pickers must see the new folder
    } catch (e) {
        console.error('Create folder error:', e);
        showToast('Error creating folder: ' + e.message, 'error');
    }
}

document.getElementById('series-name').addEventListener('input', updatePreview);
document.getElementById('tvdb-id').addEventListener('input', updatePreview);
document.getElementById('subfolder').addEventListener('input', updatePreview);

// --- 6. Shared widgets ---

// Folders known to VMSAM, fetched once and shared by every picker.
let folderCache = null;

async function fetchFolders() {
    if (folderCache) return folderCache;
    let response = null;
    try {
        response = await api.getFolders();
    } catch (e) {
        // VMSAM answers 404 while no folder is registered at all
        if (!/No folders found/i.test(e.message)) throw e;
    }
    let folders = [];
    if (Array.isArray(response)) folders = response;
    else if (response && Array.isArray(response.folders)) folders = response.folders;
    folderCache = folders;
    return folders;
}

// Server roots (CREATE_ROOT, FILES_ROOT), fetched once.
let uiConfigCache = null;

async function fetchUiConfig() {
    if (!uiConfigCache) uiConfigCache = await api.getConfig();
    return uiConfigCache;
}

// "/a//b/" -> "/a/b": paths are compared after this, never raw.
function normalizePath(path) {
    const collapsed = String(path || '').replace(/\/+/g, '/');
    return collapsed.length > 1 ? collapsed.replace(/\/$/, '') : collapsed;
}

// Absolute path under CREATE_ROOT -> path relative to it; null when outside.
function relativeToRoot(root, absolute) {
    const r = normalizePath(root);
    const a = normalizePath(absolute);
    if (a === r) return '';
    const prefix = r === '/' ? '/' : r + '/';
    return a.startsWith(prefix) ? a.slice(prefix.length) : null;
}

// A search box + dropdown over the VMSAM folders. Ids: <prefix>-folder-search,
// <prefix>-folder-list, <prefix>-selected-folder (with a <span> inside).
function createFolderPicker(prefix, onSelect, onClear) {
    const input = document.getElementById(`${prefix}-folder-search`);
    const list = document.getElementById(`${prefix}-folder-list`);
    const display = document.getElementById(`${prefix}-selected-folder`);
    const picker = { selected: null };

    const render = (query, folders) => {
        list.innerHTML = '';
        const filtered = folders.filter(f => f.destination_path.toLowerCase().includes(query.toLowerCase()));
        if (!filtered.length) list.appendChild(el('div', 'text-muted text-sm', 'No folder matches'));
        filtered.forEach(f => {
            const item = el('div', 'dir-item text-muted', f.destination_path);
            item.onclick = () => picker.select(f);
            list.appendChild(item);
        });
    };

    // Also called by code, e.g. to pick a folder that was just registered.
    picker.select = (f) => {
        picker.selected = f;
        display.querySelector('span').textContent = f.destination_path;
        display.classList.remove('hidden');
        list.classList.add('hidden');
        input.value = '';
        onSelect(f);
    };

    picker.clear = () => {
        picker.selected = null;
        display.classList.add('hidden');
        if (onClear) onClear();
    };

    picker.load = async () => {
        try {
            const folders = await fetchFolders();
            if (!folders.length) showToast('No folders found to manage', 'info');
            render(input.value, folders);
            input.oninput = (e) => render(e.target.value, folders);
        } catch (e) {
            showToast('Failed to load folders: ' + e.message, 'error');
        }
    };

    input.addEventListener('focus', () => list.classList.remove('hidden'));
    document.addEventListener('click', (e) => {
        if (!list.contains(e.target) && e.target !== input) list.classList.add('hidden');
    });
    return picker;
}

function renderRuleRow(rule, index) {
    const row = el('div', 'rule-row');
    if (index !== undefined) row.dataset.ruleIndex = String(index);
    const left = el('div', 'rule-row-patterns');
    left.appendChild(el('div', 'font-mono text-accent mb-1', rule.regex_pattern));
    left.appendChild(el('div', 'text-muted text-sm font-mono', `→ ${rule.rename_pattern || '(no rename)'}`));
    const right = el('div', 'badge', rule.weight !== undefined ? `W: ${rule.weight}` : `+${rule.episode_incremental}`);
    row.append(left, right);
    return row;
}

// Collapsible "Existing rules" panel. Returns the rules so the caller can
// test files against them. decorate(row, rule, index), optional, adds
// buttons to a rule row (Bulk Regex: "Edit regex").
async function loadExistingRulesInto(contentId, loader, emptyText, decorate) {
    const content = document.getElementById(contentId);
    content.innerHTML = '';
    content.appendChild(el('div', 'text-sm text-muted', 'Loading...'));
    try {
        const rules = await loader();
        content.innerHTML = '';
        if (!rules.length) {
            content.appendChild(el('div', 'text-sm text-muted', emptyText));
        } else {
            rules.forEach((rule, i) => {
                const row = renderRuleRow(rule, i);
                if (decorate) decorate(row, rule, i);
                content.appendChild(row);
            });
        }
        return rules;
    } catch (e) {
        content.innerHTML = '';
        content.appendChild(el('div', 'text-danger text-sm', `Failed to load: ${e.message}`));
        return [];
    }
}

async function loadFolderRules(folderId) {
    try {
        const response = await api.getRegexes(folderId);
        return response.regex_patterns || [];
    } catch (e) {
        // VMSAM answers 404 when a folder has no rule yet
        if (/No regex found/i.test(e.message)) return [];
        throw e;
    }
}

// The first rule of `rules` that extracts a valid episode number from `name`,
// as { name, rule, index, episode }, or null. Such a file needs no new rule.
function findCoveringRule(rules, name) {
    for (let i = 0; i < rules.length; i++) {
        const episode = extractEpisode(rules[i].regex_pattern, name);
        if (episode !== null && /^\d+$/.test(episode)) return { name, rule: rules[i], index: i, episode };
    }
    return null;
}

// Lists files under the rule rows of an existing-rules panel (rows built by
// loadExistingRulesInto, matched by index). entries: findCoveringRule results.
// onPropose(name), when given, adds a "Propose a new rule anyway" button.
function showRuleFiles(contentId, entries, onPropose) {
    const content = document.getElementById(contentId);
    content.querySelectorAll('.rule-files').forEach(node => node.remove());
    const byRule = new Map();
    entries.forEach(entry => {
        if (!byRule.has(entry.index)) byRule.set(entry.index, []);
        byRule.get(entry.index).push(entry);
    });
    byRule.forEach((files, index) => {
        const row = content.querySelector(`.rule-row[data-rule-index="${index}"]`);
        if (!row) return;
        const box = el('div', 'rule-files');
        box.appendChild(el('div', 'text-xs text-muted uppercase', `Already matched by this rule (${files.length})`));
        files.forEach(f => {
            const line = el('div', 'rule-file');
            const name = el('span', 'rule-file-name font-mono', f.name);
            name.title = f.name;
            line.append(name, el('span', 'badge badge-success', `ep ${f.episode}`));
            if (onPropose) {
                const propose = el('button', 'btn btn-secondary btn-sm rule-file-propose', 'Propose a new rule anyway');
                propose.type = 'button';
                propose.title = 'Put this file on the board below to give it a rule of its own';
                propose.onclick = () => onPropose(f.name);
                line.appendChild(propose);
            }
            box.appendChild(line);
        });
        row.appendChild(box);
    });
}

// Opens a collapsible section (toggleCollapsible) if it is closed.
function openCollapsible(contentId, iconId) {
    if (document.getElementById(contentId).classList.contains('hidden')) toggleCollapsible(contentId, iconId);
}

// Filter of a directory browser: shows only the listed directories whose name
// contains the text, case-insensitively (`.*text.*`, no regex). The text is
// kept while navigating, so it applies to every directory opened next.
function applyDirFilter(containerId, inputId) {
    const container = document.getElementById(containerId);
    const query = document.getElementById(inputId).value.trim().toLowerCase();
    let hidden = 0;
    container.querySelectorAll('.dir-item[data-path]').forEach(row => {
        const match = !query || (row.dataset.name || '').toLowerCase().includes(query);
        row.classList.toggle('hidden', !match);
        if (!match) hidden++;
    });
    let note = container.querySelector('.dir-filter-note');
    if (!hidden) { if (note) note.remove(); return; }
    if (!note) {
        note = el('div', 'dir-filter-note text-muted text-sm text-center p-2');
        container.appendChild(note);
    }
    const shown = container.querySelectorAll('.dir-item[data-path]:not(.hidden)').length;
    note.textContent = `${hidden} director${hidden > 1 ? 'ies' : 'y'} hidden by the filter` + (shown ? '' : ': none matches here');
}

// Large directory picker over CREATE_ROOT, a modal like the file modal (most
// of the screen, many rows visible). Used by Index Folder ("Select a
// folder…"), Move Folder and Folder Settings ("Browse…").
//   openDirectoryPicker({
//     title, hint,                 // header and help line
//     startPath,                   // directory shown first ('' = the root)
//     chooseLabel,                 // label of the button that picks the selected row
//     hereLabel, allowCurrent(p),  // optional: a button picking the directory shown
//     decorate(row, item),         // optional: badges on a row (item: { name, path })
//     onNavigate(path),            // optional: the directory shown changed
//     onPick(path)                 // a path relative to CREATE_ROOT; the picker closes
//   })
// A click selects a row, "Open →" or a double click goes inside, ".." goes up;
// the name filter (applyDirFilter) is kept while navigating. Escape, ✕,
// Cancel or a click outside close it without picking. Ids: dir-picker-*.
let dirPicker = null;

function openDirectoryPicker(options) {
    closeDirectoryPicker();
    const previousFocus = document.activeElement;
    const overlay = el('div', 'modal-overlay dir-picker-overlay');
    const box = el('div', 'modal-box file-modal-box dir-picker-box');
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-modal', 'true');
    const header = el('div', 'modal-header');
    const title = el('h3', null, options.title || 'Choose a directory');
    title.id = 'dir-picker-title';
    box.setAttribute('aria-labelledby', title.id);
    const close = el('button', 'btn-icon', '✕');
    close.type = 'button';
    close.id = 'dir-picker-close';
    close.title = 'Close without choosing';
    header.append(title, close);
    box.appendChild(header);
    if (options.hint) box.appendChild(el('p', 'text-muted text-sm dir-picker-hint', options.hint));

    const where = el('div', 'dir-picker-where');
    const whereText = el('span', 'font-mono text-accent dir-picker-path');
    whereText.id = 'dir-picker-where';
    const here = el('button', 'btn btn-secondary btn-sm', options.hereLabel || 'Choose this directory');
    here.type = 'button';
    here.id = 'dir-picker-here';
    where.append(whereText, here);
    box.appendChild(where);

    const filter = document.createElement('input');
    filter.type = 'search';
    filter.id = 'dir-picker-filter';
    filter.className = 'input dir-filter';
    filter.placeholder = 'Filter directories (name contains…)';
    filter.setAttribute('aria-label', 'Filter directories');
    filter.autocomplete = 'off';
    filter.spellcheck = false;
    box.appendChild(filter);

    const list = el('div', 'modal-body dir-browser-container dir-picker-list');
    list.id = 'dir-picker-list';
    box.appendChild(list);

    const footer = el('div', 'modal-footer dir-picker-footer');
    const selectedLine = el('span', 'text-sm dir-picker-selected-line');
    selectedLine.appendChild(el('span', 'text-muted', 'Selected: '));
    const selectedText = el('span', 'font-mono text-accent', 'none');
    selectedText.id = 'dir-picker-selected';
    selectedLine.appendChild(selectedText);
    const cancel = el('button', 'btn btn-secondary', 'Cancel');
    cancel.type = 'button';
    cancel.id = 'dir-picker-cancel';
    const choose = el('button', 'btn btn-primary', options.chooseLabel || 'Choose');
    choose.type = 'button';
    choose.id = 'dir-picker-choose';
    choose.disabled = true;
    footer.append(selectedLine, cancel, choose);
    box.appendChild(footer);
    overlay.appendChild(box);

    const picker = { overlay, options, path: '', selected: null, previousFocus, onKey: null };
    dirPicker = picker;
    const pick = (path) => {
        closeDirectoryPicker();
        options.onPick(path);
    };
    const select = (path) => {
        picker.selected = path;
        selectedText.textContent = path === null ? 'none' : (path || '/ (library root)');
        choose.disabled = path === null;
        list.querySelectorAll('.dir-item[data-path]').forEach(row => row.classList.toggle('selected', row.dataset.path === path));
    };
    const load = async (path) => {
        list.innerHTML = '';
        list.appendChild(el('div', 'text-muted text-center p-4', 'Loading...'));
        try {
            const items = await api.listFiles(path, 'create');
            if (dirPicker !== picker) return;
            picker.path = path;
            if (options.onNavigate) options.onNavigate(path);
            whereText.textContent = path ? `/${path}` : '/ (library root)';
            const allowHere = options.allowCurrent ? options.allowCurrent(path) : false;
            here.classList.toggle('hidden', !allowHere);
            list.innerHTML = '';
            if (path) {
                const up = el('div', 'dir-item text-blue dir-picker-up');
                up.tabIndex = 0;
                up.appendChild(el('span', null, '📁 ..'));
                up.onclick = () => load(path.split('/').slice(0, -1).join('/'));
                up.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); up.click(); } };
                list.appendChild(up);
            }
            const dirs = items.filter(i => i.is_dir);
            if (!dirs.length) list.appendChild(el('div', 'text-muted text-center p-2', 'No sub-folder here'));
            dirs.forEach(item => {
                const row = el('div', 'dir-item' + (picker.selected === item.path ? ' selected' : ''));
                row.dataset.path = item.path;
                row.dataset.name = item.name;
                row.tabIndex = 0;
                row.onclick = () => select(item.path);
                row.ondblclick = () => load(item.path);
                // Enter selects, Enter again chooses; → goes inside
                row.onkeydown = (e) => {
                    if (e.key === 'Enter') { e.preventDefault(); if (picker.selected === item.path) pick(item.path); else select(item.path); }
                    else if (e.key === 'ArrowRight') { e.preventDefault(); load(item.path); }
                };
                const name = el('div', 'dir-name flex-1');
                name.appendChild(el('span', null, `📁 ${item.name}`));
                const open = el('button', 'btn btn-sm btn-secondary', 'Open →');
                open.type = 'button';
                open.tabIndex = -1;
                open.onclick = (e) => { e.stopPropagation(); load(item.path); };
                const actions = el('div', 'dir-actions');
                actions.appendChild(open);
                row.append(name, actions);
                if (options.decorate) options.decorate(row, item);
                list.appendChild(row);
            });
            if (dirs.length) applyDirFilter('dir-picker-list', 'dir-picker-filter');
        } catch (e) {
            if (dirPicker !== picker) return;
            list.innerHTML = '';
            const err = el('div', 'text-danger text-sm p-4 text-center', `Error: ${e.message}`);
            if (path) { // e.g. the remembered directory was moved away
                const root = el('button', 'btn btn-sm btn-secondary mt-2', 'Back to the library root');
                root.type = 'button';
                root.onclick = () => load('');
                err.appendChild(document.createElement('br'));
                err.appendChild(root);
            }
            list.appendChild(err);
        }
    };
    picker.load = load;
    filter.addEventListener('input', () => applyDirFilter('dir-picker-list', 'dir-picker-filter'));
    here.onclick = () => pick(picker.path);
    choose.onclick = () => { if (picker.selected !== null) pick(picker.selected); };
    cancel.onclick = () => closeDirectoryPicker();
    close.onclick = () => closeDirectoryPicker();
    overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) closeDirectoryPicker(); });
    picker.onKey = (e) => {
        if (e.key === 'Escape' && !document.querySelector('.confirm-overlay')) { e.preventDefault(); e.stopPropagation(); closeDirectoryPicker(); }
    };
    document.addEventListener('keydown', picker.onKey, true);
    document.body.appendChild(overlay);
    filter.focus();
    load(options.startPath || '');
    return picker;
}

function closeDirectoryPicker() {
    if (!dirPicker) return;
    const picker = dirPicker;
    dirPicker = null;
    document.removeEventListener('keydown', picker.onKey, true);
    picker.overlay.remove();
    if (picker.previousFocus && picker.previousFocus.focus) picker.previousFocus.focus();
}

// File modal, shared: the caller says what to do with the ticked names.
let fileModalCallback = null;

async function openFileModal(onAdd) {
    fileModalCallback = onAdd;
    const modal = document.getElementById('file-modal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');

    const list = document.getElementById('file-list');
    list.innerHTML = '';
    list.appendChild(el('div', 'text-muted p-4', 'Loading files...'));
    list.onchange = updateFileSelection;
    document.getElementById('file-select-all').onclick = toggleVisibleFiles;
    updateFileSelection();

    try {
        const files = await api.listFiles('', 'files');
        const vidFiles = files.filter(f => !f.is_dir && isVideoName(f.name));
        list.innerHTML = '';
        if (!vidFiles.length) list.appendChild(el('div', 'text-muted p-4', 'No video file in the download root'));
        vidFiles.forEach(f => {
            // A <label>: a click anywhere on the row toggles its checkbox.
            const row = el('label', 'file-row');
            const check = document.createElement('input');
            check.type = 'checkbox';
            check.value = f.name;
            check.className = 'input-checkbox';
            row.title = f.name;
            row.append(check, el('span', 'text-main file-row-name', f.name));
            list.appendChild(row);
        });
        document.getElementById('file-search').value = '';
        document.getElementById('file-search').oninput = (e) => {
            const val = e.target.value;
            let regex = null;
            try { regex = val ? new RegExp(val, 'i') : null; } catch (err) { regex = null; }
            document.querySelectorAll('#file-list .file-row').forEach(row => {
                const text = row.querySelector('span').textContent;
                const match = regex ? regex.test(text) : text.toLowerCase().includes(val.toLowerCase());
                row.style.display = match ? 'flex' : 'none';
            });
            updateFileSelection();
        };
    } catch (e) {
        list.innerHTML = '';
        list.appendChild(el('div', 'text-accent p-4', e.message));
    }
    updateFileSelection();
}

// Rows the filter leaves visible: "Select all" acts on these only.
function visibleFileChecks() {
    return Array.from(document.querySelectorAll('#file-list .file-row'))
        .filter(row => row.style.display !== 'none')
        .map(row => row.querySelector('input'));
}

// Counter of every ticked file (hidden ones are added too) and the label of
// the button: "Unselect all" once every visible row is ticked.
function updateFileSelection() {
    const checked = document.querySelectorAll('#file-list input:checked').length;
    const visible = visibleFileChecks();
    const allVisibleTicked = visible.length > 0 && visible.every(c => c.checked);
    const button = document.getElementById('file-select-all');
    button.textContent = allVisibleTicked ? 'Unselect all' : 'Select all';
    button.disabled = visible.length === 0;
    document.getElementById('file-selected-count').textContent = `${checked} selected`;
}

function toggleVisibleFiles() {
    const visible = visibleFileChecks();
    const tick = !(visible.length > 0 && visible.every(c => c.checked));
    visible.forEach(c => { c.checked = tick; });
    updateFileSelection();
}

function closeFileModal() {
    document.getElementById('file-modal').classList.add('hidden');
    document.getElementById('file-modal').classList.remove('flex');
}

function addSelectedFiles() {
    const names = Array.from(document.querySelectorAll('#file-list input:checked')).map(c => c.value);
    closeFileModal();
    if (fileModalCallback) fileModalCallback(names);
}

// --- 7. Group board ---
//
// One board per tab. A group is a set of file names that one regex must
// cover; its card shows the proposal, the per-file check, and lets the user
// drag a file to another group, eject it (−), or click the digit run that is
// the episode number.

// Literal characters of a regex: an escape or a class counts once, wild
// spans, digit runs and group syntax not at all. How specific a rule is.
function regexLiteralLength(pattern) {
    return pattern
        .replace(/\(\?P?<\w+>\\d\+\)/g, '')
        .replace(/\\d\+|\.[*+]\??/g, '')
        .replace(/\[(?:\\.|[^\]\\])*\]/g, 'c')
        .replace(/\\(.)/g, '$1')
        .replace(/\(\?:|[\^$()?]/g, '').length;
}

// Where a file ejected from group `fromId` can go instead of its own card.
// A card of several files takes it when its current regex (the one shown,
// manual edit included) yields a valid episode from it; a one-file card when
// both names have the same first-pass shape (nameShape prefix: words and
// separators before the number) and the proposal for the pair yields a valid
// episode from both. Among the cards that fit, the most specific regex (most
// literal characters) wins, then the first card. { group, regex } or null.
function findEjectTarget(board, name, fromId) {
    const prefix = nameShape(name).prefix;
    let best = null;
    for (const group of board.groups) {
        if (group.id === fromId || !group.files.length || group.files.includes(name)) continue;
        let regex = null;
        if (group.files.length > 1) {
            const current = board.values(group).regex;
            if (testPattern(current, [name]).allValid) regex = current;
        } else if (nameShape(group.files[0]).prefix === prefix) {
            const proposal = analyzeGroup([group.files[0], name], group.episodeIndex);
            if (proposal.check.allValid) regex = proposal.regex;
        }
        if (regex === null) continue;
        const score = regexLiteralLength(regex);
        if (!best || score > best.score) best = { group, regex, score };
    }
    return best;
}

function createGroupBoard(options) {
    const container = document.getElementById(options.containerId);
    const board = {
        mode: options.mode, // 'regex' | 'incrementaller'
        groups: [],
        nextId: 1,
        getFolderPath: options.getFolderPath || (() => null)
    };

    board.clear = () => { board.groups = []; board.render(); };

    board.addGroups = (nameLists) => {
        nameLists.forEach(names => {
            if (!names.length) return;
            board.groups.push({ id: board.nextId++, files: [...names], episodeIndex: null, regexManual: null, renameManual: null, weight: 1, increment: 12, saved: false });
        });
        board.render();
    };

    // A replace card (Bulk Regex "Edit regex"): an existing rule whose regex
    // is edited against real files and saved with POST /regex/replace/. It
    // stays on the board without files (it cannot be saved then), and
    // Auto-group / Split all leave it alone.
    board.addReplaceGroup = (rule, names) => {
        const group = { id: board.nextId++, files: [...names], episodeIndex: null, regexManual: rule.regex_pattern, renameManual: rule.rename_pattern || '', weight: rule.weight, increment: 12, saved: false,
            replace: { regex_pattern: rule.regex_pattern, rename_pattern: rule.rename_pattern, weight: rule.weight } };
        board.groups.unshift(group);
        board.render();
        return group;
    };

    // Names from the file modal onto one card: a name on another card moves.
    board.addFilesToGroup = (id, names) => {
        const group = board.findGroup(id);
        if (!group) return;
        names.forEach(name => {
            if (group.files.includes(name)) return;
            board.groups.forEach(g => { if (g !== group && g.files.includes(name)) { g.files = g.files.filter(f => f !== name); g.saved = false; } });
            group.files.push(name);
        });
        group.saved = false;
        board.groups = board.groups.filter(g => g.files.length || g.replace);
        board.render();
    };

    board.addFiles = (names, grouped) => {
        const known = new Set(board.groups.flatMap(g => g.files));
        const fresh = names.filter(n => !known.has(n));
        if (!fresh.length) return;
        board.addGroups(grouped ? groupByNamePattern(fresh) : fresh.map(n => [n]));
    };

    // Regroup every file on the board from scratch: manual edits, chip
    // choices and saved marks of the old groups are dropped.
    board.regroup = (nameLists) => {
        board.groups = board.groups.filter(g => g.replace);
        board.addGroups(nameLists);
    };
    const looseFiles = () => board.groups.filter(g => !g.replace).flatMap(g => g.files);
    board.autoGroup = () => board.regroup(groupByNamePattern(looseFiles()));
    board.splitAll = () => board.regroup(looseFiles().map(n => [n]));

    board.findGroup = (id) => board.groups.find(g => g.id === id);

    board.moveFile = (name, fromId, toId) => {
        const from = board.findGroup(fromId);
        if (!from) return;
        if (fromId === toId) return;
        from.files = from.files.filter(f => f !== name);
        from.saved = false;
        if (toId === null) {
            board.groups.push({ id: board.nextId++, files: [name], episodeIndex: null, regexManual: null, renameManual: null, weight: from.weight, increment: from.increment, saved: false });
        } else {
            const to = board.findGroup(toId);
            if (!to) return;
            to.files.push(name);
            to.saved = false;
        }
        board.groups = board.groups.filter(g => g.files.length || g.replace);
        board.render();
    };

    // "−" on a file: another card that fits it takes it (findEjectTarget),
    // else it gets its own card. A one-file card whose manual regex would not
    // cover the pair goes back to the proposal, the one that was checked.
    board.ejectFile = (name, fromId) => {
        const target = findEjectTarget(board, name, fromId);
        if (!target) { board.moveFile(name, fromId, null); return; }
        const to = target.group;
        if (to.files.length === 1 && to.regexManual !== null && !testPattern(to.regexManual, [...to.files, name]).allValid) {
            to.regexManual = null;
            to.renameManual = null;
        }
        board.moveFile(name, fromId, to.id);
        showToast(`${name} joined the group of ${to.files[0]}`, 'success', 4000);
    };

    board.removeGroup = (id) => {
        board.groups = board.groups.filter(g => g.id !== id);
        board.render();
    };

    board.proposal = (group) => (group.files.length ? analyzeGroup(group.files, group.episodeIndex) : null);

    board.values = (group) => {
        const proposal = board.proposal(group);
        return {
            regex: group.regexManual !== null ? group.regexManual : (proposal ? proposal.regex : ''),
            rename: group.renameManual !== null ? group.renameManual : (proposal ? proposal.rename : ''),
            proposal
        };
    };

    board.render = () => {
        container.innerHTML = '';
        if (!board.groups.length) {
            container.appendChild(el('div', 'text-muted text-sm empty-board', options.emptyText || 'No file yet.'));
            return;
        }
        container.appendChild(renderBoardToolbar(board));
        board.groups.forEach(group => container.appendChild(renderGroupCard(board, group)));
        const zone = el('div', 'drop-zone', 'Drop a file here to start a new group');
        wireDropTarget(zone, board, null);
        container.appendChild(zone);
    };

    return board;
}

function renderBoardToolbar(board) {
    const fileCount = board.groups.reduce((n, g) => n + g.files.length, 0);
    const groupCount = board.groups.length;
    const bar = el('div', 'board-toolbar');
    bar.appendChild(el('span', 'text-muted text-sm board-summary',
        `${fileCount} file${fileCount > 1 ? 's' : ''} in ${groupCount} group${groupCount > 1 ? 's' : ''}`));
    const buttons = el('div', 'board-toolbar-buttons');
    const auto = el('button', 'btn btn-secondary btn-sm board-auto-group', 'Auto-group');
    auto.title = 'Regroup every file by the words its name shares with the others (manual edits are dropped)';
    auto.onclick = () => board.autoGroup();
    const split = el('button', 'btn btn-secondary btn-sm board-split-all', 'Split all');
    split.title = 'Put every file back on its own card (manual edits are dropped)';
    split.onclick = () => board.splitAll();
    buttons.append(auto, split);
    bar.appendChild(buttons);
    return bar;
}

function wireDropTarget(node, board, groupId) {
    node.addEventListener('dragover', (e) => { e.preventDefault(); node.classList.add('drag-over'); });
    node.addEventListener('dragleave', () => node.classList.remove('drag-over'));
    node.addEventListener('drop', (e) => {
        e.preventDefault();
        node.classList.remove('drag-over');
        try {
            const data = JSON.parse(e.dataTransfer.getData('text/plain'));
            board.moveFile(data.name, data.groupId, groupId);
        } catch (err) { /* not one of ours */ }
    });
}

function renderGroupCard(board, group) {
    const { regex, rename, proposal } = board.values(group);
    const replace = group.replace || null;
    const check = testPattern(regex, group.files);
    const card = el('div', `card group-card${group.saved ? ' saved' : ''}${replace ? ' replace-card' : ''}`);
    card.dataset.groupId = String(group.id);
    wireDropTarget(card, board, group.id);

    // Header
    const header = el('div', 'group-header');
    const title = el('div');
    title.appendChild(el('span', 'text-muted text-xs uppercase font-bold', replace ? 'Edit regex of an existing rule' : (board.mode === 'incrementaller' ? 'Increment rule' : 'Rule')));
    title.appendChild(el('span', 'badge', `${group.files.length} file${group.files.length === 1 ? '' : 's'}`));
    if (group.saved) title.appendChild(el('span', 'badge badge-success', replace ? 'replaced ✓' : 'saved ✓'));
    const remove = el('button', 'btn-icon', '🗑️');
    remove.title = replace ? 'Close this card (the rule is not changed)' : 'Remove this group';
    remove.onclick = () => board.removeGroup(group.id);
    header.append(title, remove);
    card.appendChild(header);

    if (replace) {
        const current = el('div', 'replace-current text-sm');
        current.appendChild(el('span', 'text-muted', 'Current regex: '));
        current.appendChild(el('code', 'font-mono text-accent replace-current-regex', replace.regex_pattern));
        card.appendChild(current);
        card.appendChild(el('div', 'text-muted text-xs replace-help', 'The new regex must match every real file of this card (they are the control); VMSAM checks it on the first one. Only the pattern changes: rename pattern and weight are changed in the Rules tab.'));
    }

    // Files
    const list = el('div', 'group-files');
    if (replace && !group.files.length) {
        list.appendChild(el('div', 'text-warning text-sm replace-no-file', 'No real file on this card: the watch folder holds no file this rule matches. Add one with "+ Add files": VMSAM needs a real file name to check the new regex on, and nothing can be saved without one.'));
    }
    group.files.forEach(name => {
        const row = el('div', 'group-file');
        row.draggable = true;
        row.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', JSON.stringify({ name, groupId: group.id }));
            e.dataTransfer.effectAllowed = 'move';
            row.classList.add('dragging');
        });
        row.addEventListener('dragend', () => row.classList.remove('dragging'));
        row.appendChild(el('span', 'drag-handle', '⠿'));
        const label = el('span', 'group-file-name font-mono', name);
        label.title = name;
        row.appendChild(label);
        const result = check.perFile.find(f => f.name === name);
        const badge = el('span', `badge ${result && result.valid ? 'badge-success' : 'badge-danger'}`, result && result.valid ? `ep ${result.episode}` : 'no match');
        row.appendChild(badge);
        if (group.files.length > 1 || replace) {
            const eject = el('button', 'btn-icon eject', '−');
            eject.title = 'Take this file out: another group whose rule fits it takes it, else it gets its own group';
            eject.onclick = () => board.ejectFile(name, group.id);
            row.appendChild(eject);
        }
        list.appendChild(row);
    });
    card.appendChild(list);

    // Reference name with clickable digit runs
    if (proposal) {
        const example = el('div', 'group-example');
        example.appendChild(el('span', 'text-muted text-xs', 'Episode number — click a digit run to change: '));
        const line = el('div', 'font-mono example-line');
        const slots = proposal.template.slots;
        const numbers = new Set(numberSlots(proposal.template));
        slots.forEach((slot, i) => {
            if (numbers.has(i)) {
                const chip = el('span', `num-chip${i === proposal.episodeIndex ? ' selected' : ''}`, slot.v);
                const candidate = proposal.candidates.find(c => c.index === i);
                chip.title = candidate ? `score ${candidate.score}: ${candidate.reasons.join(', ')}` : '';
                chip.onclick = () => { group.episodeIndex = i; group.regexManual = null; if (!replace) group.renameManual = null; group.saved = false; board.render(); };
                line.appendChild(chip);
            } else {
                line.appendChild(el('span', slot.wild ? 'wild-part' : (slot.varying ? 'varying-part' : null), slot.v));
            }
        });
        example.appendChild(line);
        card.appendChild(example);
    }

    // Inputs
    const grid = el('div', 'rule-fields');
    const regexBox = el('div');
    regexBox.appendChild(el('label', 'label', replace ? 'New Regex Pattern (Python)' : 'Regex Pattern (Python)'));
    const regexInput = document.createElement('input');
    regexInput.type = 'text';
    regexInput.className = 'input font-mono pattern-input';
    regexInput.spellcheck = false;
    regexInput.value = regex;
    regexInput.oninput = () => { group.regexManual = regexInput.value; group.saved = false; refreshCheck(); };
    regexBox.appendChild(regexInput);
    const regexMsg = el('div', 'mt-2 text-xs validation-msg');
    regexBox.appendChild(regexMsg);

    const renameBox = el('div');
    renameBox.appendChild(el('label', 'label', replace ? 'Rename Pattern (unchanged here)' : 'Rename Pattern'));
    const renameInput = document.createElement('input');
    renameInput.type = 'text';
    renameInput.className = 'input font-mono pattern-input';
    renameInput.spellcheck = false;
    renameInput.value = replace ? (replace.rename_pattern || '(no rename)') : rename;
    renameInput.disabled = !!replace;
    renameInput.oninput = () => { group.renameManual = renameInput.value; group.saved = false; refreshCheck(); };
    renameBox.appendChild(renameInput);
    const renameMsg = el('div', 'mt-2 text-xs validation-msg');
    renameBox.appendChild(renameMsg);
    grid.append(regexBox, renameBox);
    card.appendChild(grid);

    // Weight / increment + summary
    const row = el('div', 'group-footer-row');
    const numberBox = el('div');
    const numberInput = document.createElement('input');
    numberInput.type = 'number';
    numberInput.className = 'input text-center small-number';
    if (board.mode === 'incrementaller') {
        numberBox.appendChild(el('label', 'label', 'Episode increment'));
        numberInput.value = group.increment;
        numberInput.oninput = () => { group.increment = parseInt(numberInput.value, 10) || 0; group.saved = false; refreshCheck(); };
    } else {
        numberBox.appendChild(el('label', 'label', 'Weight'));
        numberInput.min = 1;
        numberInput.value = group.weight;
        numberInput.disabled = !!replace;
        numberInput.oninput = () => { group.weight = parseInt(numberInput.value, 10) || 1; group.saved = false; };
    }
    numberBox.appendChild(numberInput);
    const summary = el('div', 'extraction-box');
    row.append(numberBox, summary);
    card.appendChild(row);

    // Footer
    const footer = el('div', 'group-footer');
    if (replace) {
        const add = el('button', 'btn btn-secondary btn-sm replace-add-files', '+ Add files');
        add.type = 'button';
        add.title = 'Add real files from the watch folder: the new regex must match them too';
        add.onclick = () => openFileModal(names => board.addFilesToGroup(group.id, names));
        const reset = el('button', 'btn btn-secondary btn-sm replace-reset', 'Current regex');
        reset.type = 'button';
        reset.title = 'Put the current regex of the rule back in the field';
        reset.onclick = () => { group.regexManual = replace.regex_pattern; group.saved = false; board.render(); };
        footer.append(add, reset);
    }
    const repropose = el('button', 'btn btn-secondary btn-sm', 'Re-propose');
    repropose.title = replace ? 'A regex proposed from the files of this card' : '';
    repropose.disabled = !proposal;
    repropose.onclick = () => { group.regexManual = null; if (!replace) group.renameManual = null; group.saved = false; board.render(); };
    const save = el('button', 'btn btn-primary btn-sm group-save', replace ? 'Replace regex' : (board.mode === 'incrementaller' ? 'Save Increment Rule' : 'Save Rule'));
    save.onclick = () => saveGroup(board, group, save);
    footer.append(repropose, save);
    card.appendChild(footer);

    function refreshCheck() {
        const currentRegex = regexInput.value;
        const currentRename = renameInput.value;
        const result = testPattern(currentRegex, group.files);
        if (result.syntaxError) {
            regexMsg.textContent = '⚠ Invalid regex syntax';
            regexMsg.className = 'mt-2 text-xs text-danger validation-msg';
        } else if (replace && !group.files.length) {
            regexMsg.textContent = '⚠ No real file to check the regex on: add one';
            regexMsg.className = 'mt-2 text-xs text-warning validation-msg';
        } else if (result.allValid) {
            regexMsg.textContent = replace && currentRegex === replace.regex_pattern ? '✓ Every file matches; this is the current regex: change it to replace it' : '✓ Every file yields an episode number';
            regexMsg.className = 'mt-2 text-xs text-success validation-msg';
        } else {
            const failing = result.perFile.filter(f => !f.valid).length;
            regexMsg.textContent = `⚠ ${failing} file${failing > 1 ? 's' : ''} without a valid 'episode' group`;
            regexMsg.className = 'mt-2 text-xs text-warning validation-msg';
        }
        if (replace) {
            renameMsg.textContent = '';
        } else if (!currentRename.includes(EPISODE_PLACEHOLDER)) {
            renameMsg.textContent = `⚠ Must contain ${EPISODE_PLACEHOLDER}`;
            renameMsg.className = 'mt-2 text-xs text-warning validation-msg';
        } else {
            renameMsg.textContent = '';
            renameMsg.className = 'mt-2 text-xs text-muted validation-msg';
        }
        if (replace) {
            const problem = replaceProblem(group, currentRegex, result);
            save.disabled = !!problem;
            save.title = problem ? `Cannot replace: ${problem}` : '';
        }
        // Summary: what the reference file becomes
        summary.innerHTML = '';
        const first = result.perFile[0];
        if (first && first.valid) {
            const episode = parseInt(first.episode, 10) + (board.mode === 'incrementaller' ? group.increment : 0);
            summary.appendChild(el('span', 'text-xs text-muted uppercase', board.mode === 'incrementaller' ? 'First file becomes' : 'Extracted episode'));
            if (board.mode === 'incrementaller') {
                summary.appendChild(el('span', 'font-mono text-success preview-name', currentRename.replace(EPISODE_PLACEHOLDER, padEpisode(episode))));
            } else {
                summary.appendChild(el('span', 'text-3xl font-bold text-success', first.episode));
            }
        } else {
            summary.appendChild(el('span', 'text-xs text-muted uppercase', 'Extracted episode'));
            summary.appendChild(el('span', 'text-3xl font-bold text-muted', '-'));
        }
        // Per-file badges follow the edited regex too
        list.querySelectorAll('.group-file').forEach((rowNode, idx) => {
            const perFile = result.perFile[idx];
            const badge = rowNode.querySelector('.badge');
            badge.textContent = perFile.valid ? `ep ${perFile.episode}` : 'no match';
            badge.className = `badge ${perFile.valid ? 'badge-success' : 'badge-danger'}`;
        });
    }
    refreshCheck();
    return card;
}

// Why a replace card cannot be saved, or null.
function replaceProblem(group, regex, check) {
    if (!group.files.length) return 'no real file on the card (add one with "+ Add files")';
    if (!regex) return 'the regex is empty';
    if (check.syntaxError) return 'the regex does not compile';
    if (!check.allValid) return 'every file of the card must yield a valid episode number';
    if (regex === group.replace.regex_pattern) return 'the regex is the current one';
    return null;
}

async function saveGroup(board, group, button) {
    const { regex, rename } = board.values(group);
    const check = testPattern(regex, group.files);
    let payload;
    if (group.replace) {
        const problem = replaceProblem(group, regex, check);
        if (problem) { showToast(`Cannot replace: ${problem}`, 'error'); return; }
        payload = { regex_pattern: group.replace.regex_pattern, new_regex_pattern: regex, example_filename: group.files[0] };
    } else {
        if (!regex) { showToast('Regex pattern is required', 'error'); return; }
        if (!rename.includes(EPISODE_PLACEHOLDER)) { showToast(`Rename pattern must contain ${EPISODE_PLACEHOLDER}`, 'error'); return; }
        if (!check.allValid) { showToast('Every file of the group must yield a valid episode number', 'error'); return; }
        if (board.mode === 'incrementaller') {
            payload = { regex_pattern: regex, rename_pattern: rename, episode_incremental: group.increment, example_filename: group.files[0] };
        } else {
            const folderPath = board.getFolderPath();
            if (!folderPath) { showToast('Select a folder first', 'error'); return; }
            payload = { regex_pattern: regex, rename_pattern: rename, weight: group.weight, example_filename: group.files[0], destination_path: folderPath };
        }
    }

    button.disabled = true;
    try {
        let result;
        if (group.replace) {
            result = await api.replaceRegex(payload);
            group.replace.regex_pattern = result.regex_pattern || regex;
            group.regexManual = group.replace.regex_pattern;
        } else {
            result = board.mode === 'incrementaller' ? await api.createIncrementaller(payload) : await api.saveRegex(payload);
        }
        group.saved = true;
        const extra = result && result.new_file_name ? ` → ${result.new_file_name}`
            : (group.replace && result && result.extracted_episode !== undefined ? `: episode ${result.extracted_episode} on ${payload.example_filename}` : '');
        showToast((result && result.message ? result.message : 'Rule saved') + extra, 'success');
        board.render();
        if (board.onSaved) board.onSaved(group, result);
    } catch (e) {
        showToast((group.replace ? 'Regex not replaced: ' : 'Error saving rule: ') + e.message, 'error', 6000);
        button.disabled = false;
        return;
    }
}

// --- 8a. Bulk Regex tab ---

let regexTab = null;

function initRegexTab() {
    if (regexTab) { regexTab.picker.load(); return; }
    const board = createGroupBoard({
        containerId: 'regex-cards', mode: 'regex',
        getFolderPath: () => state.selectedFolderPath,
        emptyText: 'Add files: names that share their stable words are grouped under one rule; drag a file onto another rule to regroup.'
    });
    const picker = createFolderPicker('regex', (f) => {
        state.selectedFolderId = f.folder_id;
        state.selectedFolderPath = f.destination_path;
        document.getElementById('regex-work-area').classList.remove('disabled-area');
        document.getElementById('regex-existing').classList.remove('hidden');
        reloadRegexRules(f.folder_id);
    }, () => {
        state.selectedFolderId = null;
        state.selectedFolderPath = null;
        document.getElementById('regex-work-area').classList.add('disabled-area');
        document.getElementById('regex-existing').classList.add('hidden');
        regexTab.rules = [];
        regexTab.covered = [];
        regexTab.rulesReady = Promise.resolve([]);
        board.clear();
    });
    board.onSaved = () => reloadRegexRules(state.selectedFolderId);
    // rules: the folder's existing rules; covered: added files one of them
    // already matches, listed under it instead of on the board.
    regexTab = { board, picker, rules: [], covered: [], rulesReady: Promise.resolve([]) };
    picker.load();
    board.render();
}

// (Re)loads the existing-rules panel of folder `folderId`, then lists the
// covered files under their rule again.
function reloadRegexRules(folderId) {
    const decorate = (row, rule) => {
        const edit = el('button', 'btn btn-secondary btn-sm rule-row-edit', 'Edit regex');
        edit.type = 'button';
        edit.title = 'Edit this regex on a card holding the real files it matches';
        edit.onclick = () => startRegexReplace(rule);
        row.appendChild(edit);
    };
    const loading = loadExistingRulesInto('regex-existing-content', () => loadFolderRules(folderId), 'No existing rules for this folder.', decorate)
        .then(rules => {
            if (state.selectedFolderId === folderId) {
                regexTab.rules = rules;
                renderRegexCovered();
            }
            return rules;
        });
    regexTab.rulesReady = loading;
    return loading;
}

// Files waiting under an existing rule. A file no rule matches any more (the
// folder changed) goes back to the board.
function renderRegexCovered() {
    const entries = [];
    const lost = [];
    regexTab.covered.forEach(name => {
        const hit = findCoveringRule(regexTab.rules, name);
        if (hit) entries.push(hit); else lost.push(name);
    });
    regexTab.covered = entries.map(entry => entry.name);
    showRuleFiles('regex-existing-content', entries, proposeRegexRuleAnyway);
    if (lost.length) regexTab.board.addFiles(lost, true);
}

function proposeRegexRuleAnyway(name) {
    regexTab.covered = regexTab.covered.filter(n => n !== name);
    renderRegexCovered();
    regexTab.board.addFiles([name], true);
}

// Files from the modal: one an existing rule of the folder already matches
// is listed under that rule (no duplicate rule card), the others go to the
// board, grouped.
async function addRegexFiles(names) {
    await regexTab.rulesReady;
    const onBoard = new Set(regexTab.board.groups.flatMap(g => g.files));
    const waiting = new Set(regexTab.covered);
    const covered = [];
    const fresh = [];
    names.forEach(name => {
        if (waiting.has(name)) return;
        if (!onBoard.has(name) && findCoveringRule(regexTab.rules, name)) covered.push(name);
        else fresh.push(name);
    });
    if (covered.length) {
        regexTab.covered.push(...covered);
        renderRegexCovered();
        openCollapsible('regex-existing-content', 'regex-existing-icon');
        showToast(`${covered.length} file${covered.length > 1 ? 's' : ''} already matched by an existing rule: listed under it in Existing Rules`, 'info', 5000);
    }
    regexTab.board.addFiles(fresh, true);
}

// "Edit regex" (Existing Rules, or the Rules tab): a replace card holding the
// rule's current regex and the real files of the watch folder (top level, as
// the file modal lists it) it already matches; more can be added from the
// file modal. Files on another card stay there.
async function startRegexReplace(rule) {
    const board = regexTab.board;
    const open = board.groups.find(g => g.replace && g.replace.regex_pattern === rule.regex_pattern);
    if (open) {
        showToast('This rule is already on a card', 'info');
        scrollToGroupCard('regex-cards', open.id);
        return;
    }
    let names = [];
    try {
        names = (await api.listFiles('', 'files')).filter(f => !f.is_dir && isVideoName(f.name)).map(f => f.name);
    } catch (e) {
        showToast('Cannot list the watch folder: ' + e.message, 'error', 6000);
    }
    const matched = names.filter(name => testPattern(rule.regex_pattern, [name]).allValid);
    const onBoard = new Set(board.groups.flatMap(g => g.files));
    const files = matched.filter(name => !onBoard.has(name));
    if (files.length) {
        regexTab.covered = regexTab.covered.filter(name => !files.includes(name));
        renderRegexCovered();
    }
    const group = board.addReplaceGroup(rule, files);
    const elsewhere = matched.length - files.length;
    showToast(files.length
        ? `${files.length} real file${files.length === 1 ? '' : 's'} the rule matches put on the card` + (elsewhere ? ` (${elsewhere} already on another card)` : '')
        : 'No file of the watch folder matches this rule: add a real file to the card to check the new regex on', files.length ? 'success' : 'info', 6000);
    scrollToGroupCard('regex-cards', group.id);
}

function scrollToGroupCard(containerId, groupId) {
    const card = document.querySelector(`#${containerId} .group-card[data-group-id="${groupId}"]`);
    if (card && card.scrollIntoView) card.scrollIntoView({ block: 'center' });
}

function clearFolderSelection() {
    if (regexTab) regexTab.picker.clear();
}

function openRegexFileModal() {
    openFileModal(addRegexFiles);
}

// --- 8b. Index Folder tab ---

let indexTab = null;

function initIndexTab() {
    if (indexTab) { indexTab.picker.load(); return; }
    const board = createGroupBoard({
        containerId: 'index-cards', mode: 'regex',
        getFolderPath: () => indexTab.folder ? indexTab.folder.destination_path : null,
        emptyText: 'Every file of the folder is already indexed or covered by an existing rule.'
    });
    const picker = createFolderPicker('index', (f) => {
        indexTab.folder = f;
        indexTab.forced = new Set();
        document.getElementById('index-work-area').classList.remove('disabled-area');
        document.getElementById('index-report').classList.add('hidden');
        loadIndexFolderFiles();
    }, () => {
        indexTab.folder = null;
        document.getElementById('index-work-area').classList.add('disabled-area');
        board.clear();
    });
    // A save refreshes the rule and covered lists only: rebuilding the board
    // would throw away the edits made on the other groups.
    board.onSaved = () => loadIndexFolderFiles(false);
    // forced: covered files the user put on the board anyway ("Propose a new rule anyway")
    indexTab = { board, picker, folder: null, rules: [], forced: new Set(), registerPath: null, registerDir: '' };
    picker.load();
    board.render();
}

// After the folder cache was dropped: reload the picker and pick folder `id`
// as if the user had clicked it (its files load).
async function selectIndexFolderById(id) {
    folderCache = null;
    let folders = [];
    try {
        folders = await fetchFolders();
    } catch (e) {
        showToast('Failed to load folders: ' + e.message, 'error');
        return;
    }
    await indexTab.picker.load();
    const folder = folders.find(f => f.folder_id === id);
    if (folder) indexTab.picker.select(folder);
    else showToast(`Folder ${id} is not in the folder list`, 'error');
}

// --- Register an existing folder (a directory of CREATE_ROOT VMSAM does not know) ---

function toggleIndexRegister() {
    toggleCollapsible('index-register-content', 'index-register-icon');
}

// "Select a folder…": the library in the large directory picker; directories
// VMSAM already knows are marked "registered".
async function browseIndexRegister() {
    let config, folders;
    try {
        [config, folders] = await Promise.all([fetchUiConfig(), fetchFolders()]);
    } catch (e) {
        showToast('Cannot read the library: ' + e.message, 'error');
        return;
    }
    const registered = new Set(folders.map(f => normalizePath(f.destination_path)));
    openDirectoryPicker({
        title: 'Register an existing folder',
        hint: 'A directory of the library VMSAM does not know yet. Click a directory to select it, "Open →" or a double click to go inside. "registered" marks a directory VMSAM already knows (registering it again only updates its options).',
        startPath: indexTab.registerDir,
        chooseLabel: 'Select this folder',
        hereLabel: 'Select the directory shown',
        allowCurrent: path => !!path,
        decorate: (row, item) => {
            if (!registered.has(normalizePath(`${config.create_root}/${item.path}`))) return;
            row.classList.add('dir-registered');
            row.querySelector('.dir-name').appendChild(el('span', 'badge badge-success register-badge', 'registered'));
        },
        onNavigate: path => { indexTab.registerDir = path; },
        onPick: path => selectIndexRegisterPath(path)
    });
}

function selectIndexRegisterPath(path) {
    indexTab.registerPath = path;
    document.getElementById('index-register-path').textContent = path;
    document.getElementById('index-register-btn').disabled = false;
}

async function registerIndexFolder() {
    const path = indexTab && indexTab.registerPath;
    if (!path) { showToast('Select a folder first ("Select a folder…")', 'error'); return; }
    const number = (id, def, parse) => {
        const v = parse(document.getElementById(id).value);
        return isNaN(v) ? def : v;
    };
    const payload = {
        destination_path: path,
        original_language: document.getElementById('index-reg-lang').value.trim() || 'en',
        number_cut: number('index-reg-cut', 10, v => parseInt(v, 10)),
        cut_file_to_get_delay_second_method: number('index-reg-delay', 2.0, parseFloat),
        max_episode_number: number('index-reg-max', 12, v => parseInt(v, 10))
    };
    const button = document.getElementById('index-register-btn');
    button.disabled = true;
    try {
        const result = await api.saveFolder(payload);
        showToast(result && result.message ? result.message : 'Folder registered', 'success');
        await selectIndexFolderById(result.folder_id);
    } catch (e) {
        showToast('Cannot register the folder: ' + e.message, 'error', 6000);
    } finally {
        button.disabled = false;
    }
}

async function loadIndexFolderFiles(rebuildBoard = true) {
    const folder = indexTab.folder;
    if (!folder) return;
    const covered = document.getElementById('index-covered-content');
    covered.innerHTML = '';
    covered.appendChild(el('div', 'text-sm text-muted', 'Loading...'));
    document.getElementById('index-existing').classList.remove('hidden');

    const [rules, entries, episodes] = await Promise.all([
        loadExistingRulesInto('index-existing-content', () => loadFolderRules(folder.folder_id), 'No rule yet for this folder: create them below, then index.'),
        api.listFiles(folder.destination_path, 'create').catch(e => { showToast('Cannot list the folder: ' + e.message, 'error'); return []; }),
        api.getFolderEpisodes(folder.folder_id).then(r => (r && r.episodes) || []).catch(e => { showToast('Cannot read the indexed episodes: ' + e.message, 'error'); return []; })
    ]);
    if (indexTab.folder !== folder) return; // another folder was picked meanwhile
    indexTab.rules = rules;
    const names = entries.filter(e => !e.is_dir && isVideoName(e.name)).map(e => e.name).sort();

    // A file VMSAM already registered is done: its new name rarely matches the
    // rule that renamed it, so it must not come back as a file without rule.
    const episodeByPath = new Map(episodes.map(ep => [normalizePath(ep.file_path), ep]));
    const base = normalizePath(folder.destination_path);
    const indexedFiles = [];
    const coveredFiles = [];
    const uncovered = [];
    names.forEach(name => {
        const episode = episodeByPath.get(normalizePath(`${base}/${name}`));
        if (episode) { indexedFiles.push({ name, episode: episode.episode_number }); return; }
        // A file an existing rule already covers needs no new rule.
        // Unless the user asked for a rule of its own ("Propose a new rule anyway").
        const hit = indexTab.forced.has(name) ? null : findCoveringRule(rules, name);
        if (hit) coveredFiles.push(hit); else uncovered.push(name);
    });

    const indexedBox = document.getElementById('index-indexed-content');
    indexedBox.innerHTML = '';
    document.getElementById('index-indexed-count').textContent = `${indexedFiles.length}`;
    if (!indexedFiles.length) indexedBox.appendChild(el('div', 'text-sm text-muted', 'No file of this folder is indexed yet.'));
    indexedFiles.sort((a, b) => a.episode - b.episode).forEach(f => {
        const row = el('div', 'group-file');
        row.appendChild(el('span', 'group-file-name font-mono', f.name));
        row.appendChild(el('span', 'badge', `ep ${String(f.episode).padStart(2, '0')}`));
        indexedBox.appendChild(row);
    });

    covered.innerHTML = '';
    document.getElementById('index-covered-count').textContent = `${coveredFiles.length}`;
    if (!coveredFiles.length) covered.appendChild(el('div', 'text-sm text-muted', 'No file is covered by an existing rule yet.'));
    coveredFiles.forEach(f => {
        const row = el('div', 'group-file');
        row.appendChild(el('span', 'group-file-name font-mono', f.name));
        row.appendChild(el('span', 'badge badge-success', `ep ${f.episode}`));
        covered.appendChild(row);
    });
    // The same files, under the rule that matches each one.
    showRuleFiles('index-existing-content', coveredFiles, proposeIndexRuleAnyway);
    if (rebuildBoard && coveredFiles.length) openCollapsible('index-existing-content', 'index-existing-icon');

    document.getElementById('index-file-count').textContent =
        `${names.length} video file${names.length > 1 ? 's' : ''}: ${indexedFiles.length} indexed, ` +
        `${coveredFiles.length} covered by a rule, ${uncovered.length} without rule`;
    if (rebuildBoard) {
        indexTab.board.clear();
        indexTab.board.addGroups(groupByNamePattern(uncovered));
    }
}

// A covered file goes to the board for a rule of its own; the covered lists
// are refreshed without rebuilding the board (edits on it are kept).
function proposeIndexRuleAnyway(name) {
    indexTab.forced.add(name);
    indexTab.board.addFiles([name], true);
    loadIndexFolderFiles(false);
}

async function runIndexFolder() {
    if (!indexTab || !indexTab.folder) { showToast('Select a folder first', 'error'); return; }
    const button = document.getElementById('index-run-btn');
    button.disabled = true;
    const report = document.getElementById('index-report');
    const body = document.getElementById('index-report-content');
    try {
        const result = await api.indexFolder(indexTab.folder.folder_id);
        body.innerHTML = '';
        report.classList.remove('hidden');
        body.appendChild(el('div', 'text-success font-bold mb-4', `${result.message} — ${result.already_indexed} already in the database`));

        const table = (title, rows, columns) => {
            const box = el('div', 'mb-4');
            box.appendChild(el('h4', 'text-accent', `${title} (${rows.length})`));
            if (!rows.length) { box.appendChild(el('div', 'text-muted text-sm', 'none')); return box; }
            const t = el('table', 'report-table');
            const head = el('tr');
            columns.forEach(c => head.appendChild(el('th', null, c.label)));
            t.appendChild(head);
            rows.forEach(r => {
                const tr = el('tr');
                columns.forEach(c => tr.appendChild(el('td', c.mono ? 'font-mono' : null, c.get(r))));
                t.appendChild(tr);
            });
            box.appendChild(t);
            return box;
        };
        body.appendChild(table('Renamed and indexed', result.indexed, [
            { label: 'File', get: r => r.file_name, mono: true },
            { label: 'New name', get: r => r.new_file_name, mono: true },
            { label: 'Episode', get: r => String(r.episode_number) },
            { label: 'Weight', get: r => String(r.file_weight) }
        ]));
        body.appendChild(table('Skipped', result.skipped, [
            { label: 'File', get: r => r.file_name, mono: true },
            { label: 'Reason', get: r => r.reason.replace(/_/g, ' ') },
            { label: 'Episode', get: r => r.episode_number !== undefined ? String(r.episode_number) : (r.extracted || '') }
        ]));
        body.appendChild(table('No rule matches', result.unmatched.map(n => ({ file_name: n })), [
            { label: 'File', get: r => r.file_name, mono: true }
        ]));
        showToast(result.message, 'success');
        loadIndexFolderFiles();
    } catch (e) {
        showToast('Indexing failed: ' + e.message, 'error', 6000);
    } finally {
        button.disabled = false;
    }
}

// --- 8b'. Move Folder tab ---
//
// Moves (or renames) a registered folder: the directory on disk, the folder
// path and its episode paths, then VMSAM deletes the parent directories the
// move left empty. Paths are shown and typed relative to CREATE_ROOT; the
// relay prefixes the root and refuses "..".

let moveTab = null;

function initMoveTab() {
    if (moveTab) { moveTab.picker.load(); return; }
    // the promise is kept so a reselection after a move can wait for it
    const picker = createFolderPicker('move', (f) => { moveTab.shown = showMoveFolder(f); }, () => {
        moveTab.folder = null;
        moveTab.relative = null;
        document.getElementById('move-current').classList.add('hidden');
        document.getElementById('move-work-area').classList.add('disabled-area');
        document.getElementById('move-removed').classList.add('hidden');
        document.getElementById('move-path').value = '';
        updateMovePreview();
    });
    moveTab = { picker, folder: null, relative: null, root: null, browseDir: '', keepReport: false, shown: null };
    document.getElementById('move-path').addEventListener('input', updateMovePreview);
    picker.load();
    updateMovePreview();
}

// "/a//b/" typed by the user -> "a/b"
function cleanRelativePath(text) {
    return String(text || '').trim().replace(/\/+/g, '/').replace(/^\//, '').replace(/\/$/, '');
}

// The folder's own directory name, kept when it moves into another directory.
function moveFolderName() {
    if (!moveTab.folder) return '';
    const source = moveTab.relative !== null ? moveTab.relative : normalizePath(moveTab.folder.destination_path);
    return source.split('/').filter(Boolean).pop() || '';
}

async function showMoveFolder(folder) {
    moveTab.folder = folder;
    moveTab.relative = null;
    if (!moveTab.keepReport) document.getElementById('move-removed').classList.add('hidden');
    document.getElementById('move-work-area').classList.remove('disabled-area');
    document.getElementById('move-current').classList.remove('hidden');
    const pathNode = document.getElementById('move-current-path');
    const countNode = document.getElementById('move-episode-count');
    const input = document.getElementById('move-path');
    pathNode.textContent = folder.destination_path;
    countNode.textContent = '…';
    input.value = '';
    updateMovePreview();

    api.getFolderEpisodes(folder.folder_id)
        .then(r => {
            if (moveTab.folder !== folder) return;
            const episodes = (r && r.episodes) || [];
            countNode.textContent = String(episodes.length);
        })
        .catch(e => { if (moveTab.folder === folder) countNode.textContent = 'unknown (' + e.message + ')'; });

    try {
        const config = await fetchUiConfig();
        if (moveTab.folder !== folder) return;
        moveTab.root = config.create_root;
        document.getElementById('move-root').textContent = config.create_root;
        const relative = relativeToRoot(config.create_root, folder.destination_path);
        moveTab.relative = relative;
        if (relative === null) {
            pathNode.textContent = `${folder.destination_path} (outside the library root ${config.create_root}: the new path is taken relative to that root)`;
        } else {
            pathNode.textContent = relative || '/ (library root)';
            input.value = relative;
        }
    } catch (e) {
        showToast('Cannot read the library root: ' + e.message, 'error');
    }
    updateMovePreview();
}

function currentMoveLabel() {
    if (!moveTab.folder) return '';
    return moveTab.relative !== null ? moveTab.relative : moveTab.folder.destination_path;
}

function updateMovePreview() {
    const preview = document.getElementById('move-preview');
    const button = document.getElementById('move-btn');
    if (!moveTab || !moveTab.folder) {
        preview.textContent = 'Select a folder first';
        button.disabled = true;
        return;
    }
    const target = cleanRelativePath(document.getElementById('move-path').value);
    if (!target) {
        preview.textContent = `${currentMoveLabel()} → (enter the new path)`;
        button.disabled = true;
    } else if (target === moveTab.relative) {
        preview.textContent = `${currentMoveLabel()} → ${target} (unchanged)`;
        button.disabled = true;
    } else {
        preview.textContent = `${currentMoveLabel()} → ${target}`;
        button.disabled = false;
    }
}

// "Browse…": the library in the large directory picker. Choosing a directory
// proposes "<that directory>/<folder name>" in the free path input, which
// stays editable; the folder itself is marked (moving into it is refused).
function browseMoveDestination() {
    if (!moveTab || !moveTab.folder) { showToast('Select the folder to move first', 'error'); return; }
    const name = moveFolderName();
    openDirectoryPicker({
        title: 'Choose the destination',
        hint: `The folder goes inside the directory you choose, under its current name (${name}); the path can be edited afterwards.`,
        startPath: moveTab.browseDir,
        chooseLabel: 'Move inside the selected directory',
        hereLabel: 'Move inside the directory shown',
        allowCurrent: () => true,
        decorate: (row, item) => {
            if (moveTab.relative === null || item.path !== moveTab.relative) return;
            row.classList.add('dir-registered');
            row.querySelector('.dir-name').appendChild(el('span', 'badge badge-success register-badge', 'this folder'));
        },
        onNavigate: path => { moveTab.browseDir = path; },
        onPick: dir => setMoveTargetInside(dir)
    });
}

function setMoveTargetInside(dir) {
    if (!moveTab.folder) { showToast('Select the folder to move first', 'error'); return; }
    const name = moveFolderName();
    document.getElementById('move-path').value = dir ? `${dir}/${name}` : name;
    updateMovePreview();
}

async function moveSelectedFolder() {
    const folder = moveTab && moveTab.folder;
    if (!folder) { showToast('Select a folder first', 'error'); return; }
    const newPath = cleanRelativePath(document.getElementById('move-path').value);
    if (!newPath) { showToast('Enter the new path, relative to the library root', 'error'); return; }
    const oldLabel = currentMoveLabel();
    const confirmed = await confirmDialog(`${oldLabel}\n→ ${newPath}`, { title: 'Move this folder on disk?', okText: 'Move' });
    if (!confirmed) return;

    const button = document.getElementById('move-btn');
    const report = document.getElementById('move-removed');
    button.disabled = true;
    report.classList.add('hidden');
    try {
        const result = await api.moveFolder(folder.folder_id, newPath);
        const count = result.episodes_updated;
        showToast(`${result.message || 'Folder moved'}: ${count} episode path${count === 1 ? '' : 's'} updated`, 'success', 5000);
        showRemovedDirectories(report, result.removed_directories, moveTab.root);
        await reselectMovedFolder(folder.folder_id);
    } catch (e) {
        showToast(e.message, 'error', 8000);
    } finally {
        updateMovePreview();
    }
}

// The parent directories VMSAM deleted after a move (folders_move answers
// them absolute), shown relative to the library root in `report` and in a
// toast; `report` stays hidden when there is none.
function showRemovedDirectories(report, removedDirectories, root) {
    const removed = Array.isArray(removedDirectories) ? removedDirectories : [];
    report.innerHTML = '';
    report.classList.add('hidden');
    if (!removed.length) return;
    const shown = removed.map(p => {
        const relative = root ? relativeToRoot(root, p) : null;
        return relative === null ? p : relative;
    });
    showToast(`Empty directories deleted: ${shown.join(', ')}`, 'info', 6000);
    report.appendChild(el('div', 'text-muted', 'Empty directories deleted after the move:'));
    const list = el('ul', 'font-mono move-removed-list');
    shown.forEach(p => list.appendChild(el('li', null, p)));
    report.appendChild(list);
    report.classList.remove('hidden');
}

// A folder changed (moved, settings saved): the tabs other than `except`
// that have it selected pick it again with its new values.
function syncFolderInTabs(folder, except) {
    const id = folder.folder_id;
    if (except !== 'move' && moveTab && moveTab.folder && moveTab.folder.folder_id === id) moveTab.picker.select(folder);
    if (except !== 'index' && indexTab && indexTab.folder && indexTab.folder.folder_id === id) indexTab.picker.select(folder);
    if (except !== 'regex' && regexTab && state.selectedFolderId === id) regexTab.picker.select(folder);
    if (except !== 'episodes' && episodesTab && episodesTab.folder && episodesTab.folder.folder_id === id) episodesTab.picker.select(folder);
    if (except !== 'settings' && settingsTab && settingsTab.folder && settingsTab.folder.folder_id === id) settingsTab.picker.select(folder);
    if (except !== 'rules' && rulesTab && rulesTab.folder && rulesTab.folder.folder_id === id) rulesTab.picker.select(folder);
}

// After a move the folder keeps its id: reload the list, keep it selected in
// this tab with its new path, and refresh the other tabs that had it selected.
async function reselectMovedFolder(id) {
    folderCache = null;
    let folders = [];
    try {
        folders = await fetchFolders();
    } catch (e) {
        showToast('Failed to load folders: ' + e.message, 'error');
        return;
    }
    const folder = folders.find(f => f.folder_id === id);
    await moveTab.picker.load();
    if (!folder) { showToast(`Folder ${id} is not in the folder list`, 'error'); return; }
    moveTab.keepReport = true;
    try {
        moveTab.picker.select(folder);
        await moveTab.shown;
    } finally {
        moveTab.keepReport = false;
    }
    syncFolderInTabs(folder, 'move');
}

// --- 8c. Incrementaller tab ---

let incrementallerTab = null;

function initIncrementallerTab() {
    if (incrementallerTab) { loadIncrementallerList(); return; }
    const board = createGroupBoard({
        containerId: 'incr-cards', mode: 'incrementaller',
        emptyText: 'Add files: names that share their stable words are grouped under one rule; drag a file onto another rule to regroup.'
    });
    board.onSaved = () => loadIncrementallerList();
    incrementallerTab = { board };
    board.render();
    loadIncrementallerList();
}

function loadIncrementallerList() {
    return loadExistingRulesInto('incr-existing-content', async () => (await api.getIncrementallers()).incrementaller || [], 'No increment rule declared yet.');
}

function openIncrementallerFileModal() {
    openFileModal((names) => incrementallerTab.board.addFiles(names, false));
}

// --- 8d. Episodes tab ---
//
// The episodes VMSAM registered for a folder (GET /episodes_folder/),
// renamed in place by POST /episodes/rename/: one episode by an inline edit
// of its name, or a batch built from a name pattern holding {<episode>}.
// VMSAM checks the whole batch first and renames nothing when one name is
// refused; nothing is stored as a rule either way.

let episodesTab = null;

// "/a/b/Show A - 01.mkv" -> "Show A - 01.mkv"
function baseName(path) {
    return String(path || '').split('/').pop();
}

// "Show A - 01.mkv" -> ['Show A - 01', '.mkv']; no extension -> [name, '']
function splitExtension(name) {
    const match = String(name).match(/^(.+?)(\.[A-Za-z0-9]{1,5})$/);
    return match ? [match[1], match[2]] : [String(name), ''];
}

// The batch name of episode `number`: {<episode>} becomes the number on two
// digits at least (VMSAM writes f"{n:02}"); a pattern that does not end with
// a video extension gets the file's own extension.
function batchEpisodeName(pattern, number, currentName) {
    const name = pattern.split(EPISODE_PLACEHOLDER).join(padEpisode(number));
    return isVideoName(name) ? name : name + splitExtension(currentName)[1];
}

function initEpisodesTab() {
    if (episodesTab) { episodesTab.picker.load(); return; }
    const picker = createFolderPicker('episodes', (f) => {
        episodesTab.folder = f;
        episodesTab.selected = new Set();
        episodesTab.editing = null;
        episodesTab.patternTouched = false;
        document.getElementById('episodes-work-area').classList.remove('disabled-area');
        document.getElementById('episodes-result').classList.add('hidden');
        loadEpisodesList();
    }, () => {
        episodesTab.folder = null;
        episodesTab.episodes = [];
        episodesTab.rules = [];
        episodesTab.selected = new Set();
        episodesTab.editing = null;
        document.getElementById('episodes-work-area').classList.add('disabled-area');
        document.getElementById('episodes-result').classList.add('hidden');
        document.getElementById('episodes-count').textContent = '';
        fillEpisodesPatternChoices();
        renderEpisodesRows();
    });
    episodesTab = { picker, folder: null, episodes: [], rules: [], selected: new Set(), editing: null, patternTouched: false };
    const pattern = document.getElementById('episodes-pattern');
    pattern.addEventListener('input', () => { episodesTab.patternTouched = true; updateEpisodesPreview(); });
    document.getElementById('episodes-pattern-select').addEventListener('change', (e) => {
        if (e.target.value) { pattern.value = e.target.value; episodesTab.patternTouched = false; }
        updateEpisodesPreview();
    });
    document.getElementById('episodes-select-all').onclick = toggleAllEpisodes;
    document.getElementById('episodes-batch-btn').onclick = renameEpisodesBatch;
    picker.load();
    fillEpisodesPatternChoices();
    renderEpisodesRows();
}

async function loadEpisodesList() {
    const folder = episodesTab && episodesTab.folder;
    if (!folder) return;
    const count = document.getElementById('episodes-count');
    count.textContent = 'Loading…';
    try {
        const [response, rules] = await Promise.all([
            api.getFolderEpisodes(folder.folder_id),
            loadFolderRules(folder.folder_id).catch(() => [])
        ]);
        if (episodesTab.folder !== folder) return;
        episodesTab.episodes = ((response && response.episodes) || []).slice().sort((a, b) => a.episode_number - b.episode_number);
        episodesTab.rules = rules;
        const known = new Set(episodesTab.episodes.map(ep => ep.episode_number));
        episodesTab.selected = new Set(Array.from(episodesTab.selected).filter(n => known.has(n)));
        const n = episodesTab.episodes.length;
        count.textContent = `${n} registered episode${n === 1 ? '' : 's'}`;
    } catch (e) {
        if (episodesTab.folder !== folder) return;
        episodesTab.episodes = [];
        count.textContent = `Failed to load: ${e.message}`;
        showToast('Failed to load the episodes: ' + e.message, 'error');
    }
    fillEpisodesPatternChoices();
    renderEpisodesRows();
}

// "Start from": the rename patterns of the folder's rules, then a pattern
// proposed from the current file names. The first one fills the pattern box
// until the user types in it.
function fillEpisodesPatternChoices() {
    const select = document.getElementById('episodes-pattern-select');
    const input = document.getElementById('episodes-pattern');
    const choices = [];
    const add = (value, label) => { if (value && !choices.some(c => c.value === value)) choices.push({ value, label }); };
    (episodesTab ? episodesTab.rules : []).forEach(rule => add(rule.rename_pattern, `Rule rename pattern: ${rule.rename_pattern}`));
    const names = (episodesTab ? episodesTab.episodes : []).map(ep => baseName(ep.file_path));
    if (names.length) {
        try {
            const proposal = analyzeGroup(names).rename;
            if (proposal.includes(EPISODE_PLACEHOLDER)) add(proposal, `Proposed from the current names: ${proposal}`);
        } catch (e) { /* no proposal */ }
    }
    select.innerHTML = '';
    const first = document.createElement('option');
    first.value = '';
    first.textContent = choices.length ? '— type a pattern, or pick one —' : 'No rename pattern in this folder: type one';
    select.appendChild(first);
    choices.forEach(choice => {
        const option = document.createElement('option');
        option.value = choice.value;
        option.textContent = choice.label;
        select.appendChild(option);
    });
    if (episodesTab && !episodesTab.patternTouched) {
        input.value = choices.length ? choices[0].value : '';
        select.value = input.value;
    }
}

function renderEpisodesRows() {
    const body = document.getElementById('episodes-rows');
    body.innerHTML = '';
    const episodes = episodesTab ? episodesTab.episodes : [];
    if (!episodes.length) {
        const row = el('tr');
        const cell = el('td', 'text-muted', episodesTab && episodesTab.folder ? 'No episode registered for this folder.' : 'Select a folder first.');
        cell.colSpan = 6;
        row.appendChild(cell);
        body.appendChild(row);
    }
    episodes.forEach(ep => {
        const number = ep.episode_number;
        const name = baseName(ep.file_path);
        const row = el('tr', 'eps-row');
        row.dataset.episode = String(number);

        const checkCell = el('td');
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.className = 'input-checkbox eps-check';
        box.checked = episodesTab.selected.has(number);
        box.setAttribute('aria-label', `Select episode ${number}`);
        box.onchange = () => {
            if (box.checked) episodesTab.selected.add(number); else episodesTab.selected.delete(number);
            updateEpisodesPreview();
        };
        checkCell.appendChild(box);

        const nameCell = el('td', 'font-mono eps-name');
        const actionCell = el('td', 'eps-actions');
        if (episodesTab.editing === number) {
            const [stem, extension] = splitExtension(name);
            const edit = el('div', 'eps-edit');
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'input font-mono eps-edit-input';
            input.spellcheck = false;
            input.value = stem;
            input.setAttribute('aria-label', `New name of episode ${number}`);
            const save = el('button', 'btn btn-primary btn-sm eps-save', 'Save');
            const cancel = el('button', 'btn btn-secondary btn-sm eps-cancel', 'Cancel');
            save.type = 'button';
            cancel.type = 'button';
            save.onclick = () => renameOneEpisode(ep, input.value.trim() + extension, save);
            cancel.onclick = () => { episodesTab.editing = null; renderEpisodesRows(); };
            input.onkeydown = (e) => {
                if (e.key === 'Enter') { e.preventDefault(); save.click(); }
                else if (e.key === 'Escape') { e.preventDefault(); cancel.click(); }
            };
            edit.append(input, el('span', 'eps-ext font-mono', extension || '(no extension)'));
            nameCell.appendChild(edit);
            actionCell.append(save, cancel);
            setTimeout(() => input.focus(), 0);
        } else {
            nameCell.textContent = name;
            nameCell.title = ep.file_path;
            const rename = el('button', 'btn btn-secondary btn-sm eps-rename', 'Rename');
            rename.type = 'button';
            rename.onclick = () => { episodesTab.editing = number; renderEpisodesRows(); };
            actionCell.appendChild(rename);
        }

        row.append(checkCell, el('td', 'font-mono', padEpisode(number)), nameCell,
            el('td', 'font-mono', ep.file_weight === null || ep.file_weight === undefined ? '-' : String(ep.file_weight)),
            el('td', 'font-mono eps-preview'), actionCell);
        body.appendChild(row);
    });
    updateEpisodesPreview();
}

function toggleAllEpisodes() {
    const episodes = episodesTab.episodes;
    const all = episodes.length && episodes.every(ep => episodesTab.selected.has(ep.episode_number));
    episodesTab.selected = all ? new Set() : new Set(episodes.map(ep => ep.episode_number));
    document.querySelectorAll('#episodes-rows .eps-row').forEach(row => {
        row.querySelector('.eps-check').checked = episodesTab.selected.has(parseInt(row.dataset.episode, 10));
    });
    updateEpisodesPreview();
}

// The batch as [{ ep, oldName, newName }], selected episodes in number order.
function episodesBatchPlan() {
    const pattern = document.getElementById('episodes-pattern').value;
    return episodesTab.episodes
        .filter(ep => episodesTab.selected.has(ep.episode_number))
        .map(ep => {
            const oldName = baseName(ep.file_path);
            return { ep, oldName, newName: batchEpisodeName(pattern, ep.episode_number, oldName) };
        });
}

// Live preview of the batch in the "Preview" column, the checks VMSAM would
// refuse anyway (placeholder, path, two files on one name), and the button.
function updateEpisodesPreview() {
    if (!episodesTab) return;
    const pattern = document.getElementById('episodes-pattern').value;
    const msg = document.getElementById('episodes-pattern-msg');
    const button = document.getElementById('episodes-batch-btn');
    const selectAll = document.getElementById('episodes-select-all');
    const episodes = episodesTab.episodes;
    const selectedCount = episodes.filter(ep => episodesTab.selected.has(ep.episode_number)).length;
    document.getElementById('episodes-selected-count').textContent = `${selectedCount} selected`;
    selectAll.textContent = episodes.length && selectedCount === episodes.length ? 'Unselect all' : 'Select all';

    let problem = null;
    if (!pattern.trim()) problem = 'Enter a name pattern';
    else if (!pattern.includes(EPISODE_PLACEHOLDER)) problem = `The pattern must contain ${EPISODE_PLACEHOLDER}`;
    else if (/[\/\\]/.test(pattern)) problem = 'A file name pattern, not a path';

    const plan = problem ? [] : episodesBatchPlan();
    const byEpisode = new Map(plan.map(p => [p.ep.episode_number, p]));
    // a target that is another file of the folder, or the target of two episodes
    const targets = new Map();
    plan.forEach(p => targets.set(p.newName, (targets.get(p.newName) || 0) + 1));
    const others = new Set(episodes.filter(ep => !byEpisode.has(ep.episode_number)).map(ep => baseName(ep.file_path)));
    let conflicts = 0;
    let changed = 0;
    document.querySelectorAll('#episodes-rows .eps-row').forEach(row => {
        const cell = row.querySelector('.eps-preview');
        const p = byEpisode.get(parseInt(row.dataset.episode, 10));
        cell.textContent = '';
        cell.className = 'font-mono eps-preview';
        if (!p) return;
        if (p.newName === p.oldName) {
            cell.textContent = '= unchanged';
            cell.classList.add('text-muted');
        } else if (targets.get(p.newName) > 1 || others.has(p.newName)) {
            conflicts++;
            cell.textContent = `→ ${p.newName} (already taken)`;
            cell.classList.add('text-danger');
        } else {
            changed++;
            cell.textContent = `→ ${p.newName}`;
            cell.classList.add('text-success');
        }
    });

    if (!problem && conflicts) problem = `${conflicts} new name${conflicts > 1 ? 's are' : ' is'} already taken: change the pattern or the selection`;
    if (problem) {
        msg.textContent = `⚠ ${problem}`;
        msg.className = 'text-xs validation-msg text-warning';
    } else if (!selectedCount) {
        msg.textContent = 'Tick the episodes to rename.';
        msg.className = 'text-xs validation-msg text-muted';
    } else {
        msg.textContent = `${changed} file${changed === 1 ? '' : 's'} to rename` + (selectedCount > changed ? `, ${selectedCount - changed} unchanged` : '');
        msg.className = 'text-xs validation-msg text-success';
    }
    button.disabled = !!problem || !changed;
    button.textContent = changed ? `Rename ${changed} episode${changed === 1 ? '' : 's'}` : 'Rename selected';
}

// VMSAM's answer, kept on screen: what was renamed, or the reason nothing was.
function showEpisodesResult(result, refusal) {
    const box = document.getElementById('episodes-result');
    box.innerHTML = '';
    box.classList.remove('hidden');
    if (refusal) {
        box.className = 'text-sm mt-2 eps-result text-danger';
        box.textContent = `VMSAM refused the rename: ${refusal}`;
        showToast(refusal, 'error', 8000);
        return;
    }
    box.className = 'text-sm mt-2 eps-result';
    box.appendChild(el('div', 'text-success', result.message || 'Renamed'));
    const list = el('ul', 'font-mono eps-result-list');
    (result.renamed || []).forEach(r => {
        list.appendChild(el('li', null, `${padEpisode(r.episode_number)}: ${r.old_file_name} → ${r.new_file_name}`));
    });
    box.appendChild(list);
    showToast(result.message || 'Renamed', 'success');
}

async function renameOneEpisode(ep, newName, button) {
    const folder = episodesTab.folder;
    const oldName = baseName(ep.file_path);
    if (!newName || newName === oldName) { showToast('Enter a new name first', 'error'); return; }
    if (/[\/\\]/.test(newName)) { showToast('A file name, not a path', 'error'); return; }
    button.disabled = true;
    try {
        const result = await api.renameEpisodes(folder.folder_id, [{ episode_number: ep.episode_number, new_file_name: newName }]);
        episodesTab.editing = null;
        showEpisodesResult(result);
        await loadEpisodesList();
    } catch (e) {
        showEpisodesResult(null, e.message);
        button.disabled = false;
    }
}

async function renameEpisodesBatch() {
    const folder = episodesTab.folder;
    if (!folder) return;
    updateEpisodesPreview();
    if (document.getElementById('episodes-batch-btn').disabled) return;
    const plan = episodesBatchPlan().filter(p => p.newName !== p.oldName);
    const shown = plan.slice(0, 12).map(p => `${padEpisode(p.ep.episode_number)}: ${p.oldName}\n    → ${p.newName}`);
    if (plan.length > shown.length) shown.push(`… and ${plan.length - shown.length} more`);
    const confirmed = await confirmDialog(shown.join('\n'), { title: `Rename ${plan.length} episode file${plan.length === 1 ? '' : 's'} on disk?`, okText: 'Rename' });
    if (!confirmed) return;
    const button = document.getElementById('episodes-batch-btn');
    button.disabled = true;
    try {
        const result = await api.renameEpisodes(folder.folder_id, plan.map(p => ({ episode_number: p.ep.episode_number, new_file_name: p.newName })));
        episodesTab.selected = new Set();
        showEpisodesResult(result);
        await loadEpisodesList();
    } catch (e) {
        showEpisodesResult(null, e.message);
        updateEpisodesPreview();
    }
}

// --- 8e. Folder Settings tab ---
//
// The parameters and the path of a registered folder. POST /folder/ creates
// or updates the folder whose destination_path it is given, with the FULL
// data, so the path is the key: a new path is first a move (POST
// /folders/move/, confirmed, the removed directories shown), then the
// parameters are saved under the new path. An unchanged path is only the save.

let settingsTab = null;

const SETTINGS_FIELDS = [
    { key: 'original_language', id: 'settings-lang', label: 'Original Language', kind: 'text' },
    { key: 'number_cut', id: 'settings-cut', label: 'Number Cut', kind: 'int' },
    { key: 'max_episode_number', id: 'settings-max', label: 'Max Episode', kind: 'int' },
    { key: 'cut_file_to_get_delay_second_method', id: 'settings-delay', label: 'Cut Ratio', kind: 'ratio' }
];

function initSettingsTab() {
    if (settingsTab) { settingsTab.picker.load(); return; }
    // the promise is kept so a reselection after a save can wait for it
    const picker = createFolderPicker('settings', (f) => { settingsTab.shown = showSettingsFolder(f); }, () => {
        settingsTab.folder = null;
        document.getElementById('settings-work-area').classList.add('disabled-area');
        document.getElementById('settings-removed').classList.add('hidden');
        fillSettingsForm(null);
    });
    settingsTab = { picker, folder: null, root: null, browseDir: '', keepReport: false, shown: null };
    SETTINGS_FIELDS.forEach(field => document.getElementById(field.id).addEventListener('input', validateSettingsForm));
    document.getElementById('settings-path').addEventListener('input', validateSettingsForm);
    document.getElementById('settings-save-btn').onclick = saveFolderSettings;
    document.getElementById('settings-reset-btn').onclick = () => fillSettingsForm(settingsTab.folder);
    document.getElementById('settings-browse-btn').onclick = browseSettingsPath;
    picker.load();
    fillSettingsForm(null);
}

// Shows the folder of the list at once, then VMSAM's current values.
async function showSettingsFolder(folder) {
    settingsTab.folder = folder;
    if (!settingsTab.keepReport) document.getElementById('settings-removed').classList.add('hidden');
    document.getElementById('settings-work-area').classList.remove('disabled-area');
    fillSettingsForm(folder);
    try {
        const [fresh, config] = await Promise.all([api.getFolder(folder.folder_id), fetchUiConfig()]);
        if (!settingsTab.folder || settingsTab.folder.folder_id !== folder.folder_id) return;
        settingsTab.root = config.create_root;
        document.getElementById('settings-root').textContent = config.create_root;
        settingsTab.folder = fresh;
        fillSettingsForm(fresh);
    } catch (e) {
        showToast('Failed to read the folder: ' + e.message, 'error');
    }
}

// The folder path relative to the library root, as the relay rebuilds it
// (CREATE_ROOT + "/" + relative), or null: outside the root, the root
// itself, or stored in another spelling (`//`, a trailing `/`). POST /folder/
// is keyed by the path, so saving such a folder would create another one.
function settingsRelative(folder) {
    if (!folder || !settingsTab || !settingsTab.root) return null;
    const relative = relativeToRoot(settingsTab.root, folder.destination_path);
    if (!relative) return null;
    const root = normalizePath(settingsTab.root);
    return (root === '/' ? '/' : root + '/') + relative === folder.destination_path ? relative : null;
}

function fillSettingsForm(folder) {
    const path = document.getElementById('settings-path');
    const relative = settingsRelative(folder);
    path.value = folder ? (relative !== null ? relative : folder.destination_path) : '';
    path.disabled = !folder || relative === null;
    document.getElementById('settings-browse-btn').disabled = path.disabled;
    let current = '';
    if (folder) {
        current = `current: ${relative !== null ? relative : folder.destination_path}`;
        if (relative === null && settingsTab && settingsTab.root) current += ' (outside the library root or not written as root + relative path: this tab cannot save it)';
    }
    document.getElementById('settings-path-current').textContent = current;
    SETTINGS_FIELDS.forEach(field => {
        const value = folder ? folder[field.key] : null;
        document.getElementById(field.id).value = value === null || value === undefined ? '' : String(value);
        document.getElementById(`${field.id}-current`).textContent = folder ? `current: ${value}` : '';
    });
    validateSettingsForm();
}

// { values: every field, changed: the fields that differ, newPath: the new
//   relative path or null when unchanged, relative: the current one, errors }
function readSettingsForm() {
    const folder = settingsTab.folder;
    const values = {};
    const changed = {};
    const errors = [];
    SETTINGS_FIELDS.forEach(field => {
        const raw = document.getElementById(field.id).value.trim();
        let value;
        if (field.kind === 'text') {
            if (!raw) { errors.push(`${field.label} cannot be empty`); return; }
            value = raw;
        } else {
            value = Number(raw);
            if (!raw || !Number.isFinite(value)) { errors.push(`${field.label} must be a number`); return; }
            if (field.kind === 'int' && (!Number.isInteger(value) || value < 1)) { errors.push(`${field.label} must be a whole number of 1 or more`); return; }
            if (field.kind === 'ratio' && value <= 1) { errors.push(`${field.label} must be greater than 1`); return; }
        }
        values[field.key] = value;
        if (folder && value !== folder[field.key]) changed[field.key] = value;
    });
    const relative = settingsRelative(folder);
    let newPath = null;
    if (folder && relative === null) {
        errors.push('the folder path is outside the library root (or not written as root + relative path): nothing can be saved from this tab');
    } else if (folder) {
        const typed = cleanRelativePath(document.getElementById('settings-path').value);
        if (!typed) errors.push('The destination path cannot be empty');
        else if (typed.split('/').includes('..')) errors.push('The destination path cannot hold ".."');
        else if (typed !== relative) newPath = typed;
    }
    return { values, changed, newPath, relative, errors };
}

function validateSettingsForm() {
    const msg = document.getElementById('settings-msg');
    const save = document.getElementById('settings-save-btn');
    if (!settingsTab || !settingsTab.folder) { msg.textContent = ''; save.disabled = true; save.textContent = 'Save'; return; }
    const { changed, newPath, relative, errors } = readSettingsForm();
    const count = Object.keys(changed).length;
    if (errors.length) {
        msg.textContent = '⚠ ' + errors.join('; ');
        msg.className = 'text-xs validation-msg text-warning';
    } else {
        const parts = [];
        if (newPath) parts.push(`moved first: ${relative} → ${newPath}`);
        if (count) parts.push(`${count} setting${count === 1 ? '' : 's'} changed`);
        msg.textContent = parts.length ? parts.join('; ') : 'No change';
        msg.className = `text-xs validation-msg ${parts.length ? 'text-success' : 'text-muted'}`;
    }
    save.disabled = !!errors.length || (!count && !newPath);
    save.textContent = newPath ? (count ? 'Move and save' : 'Move') : 'Save';
}

// "Browse…": choosing a directory proposes "<directory>/<folder name>".
function browseSettingsPath() {
    const folder = settingsTab && settingsTab.folder;
    const relative = settingsRelative(folder);
    if (relative === null) return;
    const name = relative.split('/').pop();
    openDirectoryPicker({
        title: 'Choose the new location',
        hint: `The folder goes inside the directory you choose, under its current name (${name}); the path can be edited afterwards. Saving moves it on disk first.`,
        startPath: settingsTab.browseDir,
        chooseLabel: 'Inside the selected directory',
        hereLabel: 'Inside the directory shown',
        allowCurrent: () => true,
        decorate: (row, item) => {
            if (item.path !== relative) return;
            row.classList.add('dir-registered');
            row.querySelector('.dir-name').appendChild(el('span', 'badge badge-success register-badge', 'this folder'));
        },
        onNavigate: path => { settingsTab.browseDir = path; },
        onPick: dir => {
            document.getElementById('settings-path').value = dir ? `${dir}/${name}` : name;
            validateSettingsForm();
        }
    });
}

async function saveFolderSettings() {
    const folder = settingsTab.folder;
    if (!folder) return;
    const { values, changed, newPath, relative, errors } = readSettingsForm();
    const count = Object.keys(changed).length;
    if (errors.length || (!count && !newPath)) { validateSettingsForm(); return; }
    const msg = document.getElementById('settings-msg');
    const refused = (text) => {
        msg.textContent = text;
        msg.className = 'text-xs validation-msg text-danger';
    };
    if (newPath) {
        const then = count ? `\n\nThen ${count} changed setting${count === 1 ? ' is' : 's are'} saved.` : '';
        const confirmed = await confirmDialog(`${relative}\n→ ${newPath}${then}`, { title: 'Move this folder on disk?', okText: count ? 'Move and save' : 'Move' });
        if (!confirmed) return;
    }
    const save = document.getElementById('settings-save-btn');
    save.disabled = true;
    const report = document.getElementById('settings-removed');
    let path = relative;
    if (newPath) {
        try {
            const result = await api.moveFolder(folder.folder_id, newPath);
            const moved = result.episodes_updated;
            showToast(`${result.message || 'Folder moved'}: ${moved} episode path${moved === 1 ? '' : 's'} updated`, 'success', 5000);
            showRemovedDirectories(report, result.removed_directories, settingsTab.root);
            path = newPath;
        } catch (e) {
            showToast('Folder not moved: ' + e.message, 'error', 8000);
            refused(`VMSAM refused the move, nothing saved: ${e.message}`);
            save.disabled = false;
            return;
        }
    } else {
        report.classList.add('hidden');
    }
    let saveError = null;
    if (count) {
        try {
            const result = await api.saveFolder(Object.assign({ destination_path: path }, values));
            if (result.folder_id !== folder.folder_id) {
                saveError = `VMSAM answered "${result.message}" for folder ${result.folder_id}, not folder ${folder.folder_id}: check the folder list`;
                showToast(saveError, 'error', 10000);
            } else {
                showToast(result.message || 'Folder updated', 'success');
            }
        } catch (e) {
            saveError = e.message;
            showToast('Settings not saved: ' + e.message, 'error', 8000);
        }
    }
    // reload the folder everywhere: its path may have changed even if the save failed
    folderCache = null;
    let updated = null;
    try {
        updated = (await fetchFolders()).find(f => f.folder_id === folder.folder_id) || null;
    } catch (e) {
        showToast('Failed to load folders: ' + e.message, 'error');
    }
    await settingsTab.picker.load();
    if (updated && settingsTab.folder && settingsTab.folder.folder_id === folder.folder_id) {
        settingsTab.keepReport = true;
        try {
            settingsTab.picker.select(updated);
            await settingsTab.shown;
        } finally {
            settingsTab.keepReport = false;
        }
        syncFolderInTabs(updated, 'settings');
    }
    if (saveError) refused(`VMSAM refused the settings${newPath ? ' (the folder was moved)' : ''}: ${saveError}`);
}

// --- 8f. Rules tab ---
//
// The rules of a folder: weight, destination folder and rename pattern
// changed through POST /regex/ (an existing pattern is updated, with the
// FULL data: same regex_pattern, rename_pattern, weight, the chosen folder's
// destination_path and an example_filename the regex matches), or deleted
// (POST /regex/delete/). The example is built here (proposeRuleExample);
// when nothing built matches, the user gives a real file name. The regex is
// read-only: "Edit regex in Bulk Regex" edits it against real files.

let rulesTab = null;

function initRulesTab() {
    if (rulesTab) { rulesTab.picker.load(); if (rulesTab.folder) loadRulesList(); return; }
    const picker = createFolderPicker('rules', (f) => {
        rulesTab.folder = f;
        document.getElementById('rules-work-area').classList.remove('disabled-area');
        loadRulesList();
    }, () => {
        rulesTab.folder = null;
        document.getElementById('rules-work-area').classList.add('disabled-area');
        document.getElementById('rules-cards').innerHTML = '';
        document.getElementById('rules-count').textContent = '';
    });
    rulesTab = { picker, folder: null };
    picker.load();
}

async function loadRulesList() {
    const folder = rulesTab && rulesTab.folder;
    if (!folder) return;
    const container = document.getElementById('rules-cards');
    container.innerHTML = '';
    container.appendChild(el('div', 'text-muted text-sm', 'Loading...'));
    let rules = [];
    let folders = [];
    try {
        [rules, folders] = await Promise.all([loadFolderRules(folder.folder_id), fetchFolders()]);
    } catch (e) {
        if (rulesTab.folder !== folder) return;
        container.innerHTML = '';
        container.appendChild(el('div', 'text-danger text-sm', `Failed to load: ${e.message}`));
        return;
    }
    if (rulesTab.folder !== folder) return;
    container.innerHTML = '';
    document.getElementById('rules-count').textContent = `(${rules.length})`;
    if (!rules.length) container.appendChild(el('div', 'card text-muted text-sm', 'No rule for this folder.'));
    const sortedFolders = folders.slice().sort((a, b) => a.destination_path.localeCompare(b.destination_path));
    rules.forEach(rule => container.appendChild(renderRuleEditor(rule, sortedFolders, folder)));
}

// The rename pattern with episode 01: what the rule's renamed files look like.
function ruleExampleName(renamePattern) {
    return renamePattern ? renamePattern.split(EPISODE_PLACEHOLDER).join(padEpisode(1)) : '';
}

// A file name a regex matches, built from the pattern itself: the episode
// group as 01, \d as 1, a wild `.` as "x", a character class as its first
// character, an optional `(?:…)?` group dropped, escaped characters as
// themselves. Best effort: null on any other construct (alternation, a
// quantified literal, \w…); callers keep it only when testPattern accepts it.
function sampleFromRegex(pattern) {
    const text = String(pattern || '');
    // length of the quantifier at `at` (+, *, ?, {n,m}, lazy or not), and its minimum count
    const quantifier = (at) => {
        const match = text.slice(at).match(/^(?:([+*?])|\{(\d+)(?:,\d*)?\})\??/);
        if (!match) return { length: 0, min: 1 };
        return { length: match[0].length, min: match[2] !== undefined ? parseInt(match[2], 10) : (match[1] === '+' ? 1 : 0) };
    };
    // index of the `)` closing the group opened at `at`, escapes and classes skipped
    const closing = (at) => {
        let depth = 0;
        for (let k = at; k < text.length; k++) {
            if (text[k] === '\\') { k++; continue; }
            if (text[k] === '(') depth++;
            else if (text[k] === ')' && --depth === 0) return k;
        }
        return -1;
    };
    let out = '';
    let i = text.startsWith('^') ? 1 : 0;
    while (i < text.length) {
        const c = text[i];
        if (c === '$' && i === text.length - 1) break;
        if (c === '\\') {
            const next = text[i + 1];
            if (next === 'd') {
                const q = quantifier(i + 2);
                out += '1'.repeat(Math.max(1, q.min));
                i += 2 + q.length;
                continue;
            }
            if (next === undefined || /[A-Za-z0-9]/.test(next)) return null;
            out += next;
            i += 2;
            continue;
        }
        if (text.startsWith('(?P<episode>', i)) {
            const end = closing(i);
            if (end < 0) return null;
            out += padEpisode(1);
            i = end + 1;
            continue;
        }
        if (text.startsWith('(?:', i)) {
            const end = closing(i);
            if (end < 0 || text[end + 1] !== '?') return null;
            i = end + 2;
            continue;
        }
        if (c === '[') {
            const end = text.indexOf(']', i + 2);
            const first = text[i + 1];
            if (end < 0 || first === '^' || first === '\\') return null;
            out += first;
            i = end + 1 + quantifier(end + 1).length;
            continue;
        }
        if (c === '.') {
            out += 'x';
            i += 1 + quantifier(i + 1).length;
            continue;
        }
        if ('()|*+?{}^$'.includes(c)) return null;
        out += c;
        i++;
    }
    return out;
}

// The example file name sent with a rule: a rename pattern with episode 01
// when the regex matches it (the edited pattern first, then the stored one),
// else a name built from the regex (sampleFromRegex); null when none of them
// yields a valid episode, and the user has to give a real file name.
function proposeRuleExample(regex, ...renames) {
    const candidates = [...renames.map(ruleExampleName), sampleFromRegex(regex)].filter(Boolean);
    return candidates.find(name => testPattern(regex, [name]).allValid) || null;
}

function renderRuleEditor(rule, folders, folder) {
    const card = el('div', 'card group-card rul-card');
    card.dataset.pattern = rule.regex_pattern;

    const header = el('div', 'group-header');
    header.appendChild(el('span', 'text-muted text-xs uppercase font-bold', 'Rule'));
    header.appendChild(el('span', 'badge', `W: ${rule.weight}`));
    card.appendChild(header);

    // the regex, read-only: it is edited in Bulk Regex, against real files
    const regexBox = el('div', 'rul-regex-box');
    regexBox.appendChild(el('label', 'label', 'Regex Pattern (Python) — read-only here'));
    const regexLine = el('div', 'flex items-center rul-regex-row');
    const regexText = el('code', 'font-mono text-accent rul-regex', rule.regex_pattern);
    const editRegex = el('button', 'btn btn-secondary btn-sm rul-edit-regex', 'Edit regex in Bulk Regex');
    editRegex.type = 'button';
    editRegex.title = 'Opens Bulk Regex on this folder with this rule on a card, with the real files it matches';
    editRegex.onclick = () => openRegexEdit(folder, rule);
    regexLine.append(regexText, editRegex);
    regexBox.appendChild(regexLine);
    card.appendChild(regexBox);

    const renameBox = el('div', 'mt-4');
    renameBox.appendChild(el('label', 'label', 'Rename Pattern (empty: the file keeps its name)'));
    const rename = document.createElement('input');
    rename.type = 'text';
    rename.className = 'input font-mono pattern-input rul-rename';
    rename.spellcheck = false;
    rename.autocomplete = 'off';
    rename.value = rule.rename_pattern || '';
    const renameMsg = el('div', 'mt-1 text-xs validation-msg');
    renameBox.append(rename, renameMsg);
    card.appendChild(renameBox);

    const row = el('div', 'rul-row mt-4');
    const weightBox = el('div');
    weightBox.appendChild(el('label', 'label', 'Weight'));
    const weight = document.createElement('input');
    weight.type = 'number';
    weight.min = 1;
    weight.step = 1;
    weight.className = 'input text-center small-number rul-weight';
    weight.value = rule.weight;
    weightBox.appendChild(weight);
    const folderBox = el('div', 'rul-folder-box');
    folderBox.appendChild(el('label', 'label', 'Destination folder'));
    const folderSelect = document.createElement('select');
    folderSelect.className = 'input font-mono rul-folder';
    folders.forEach(f => {
        const option = document.createElement('option');
        option.value = String(f.folder_id);
        option.textContent = f.destination_path;
        folderSelect.appendChild(option);
    });
    folderSelect.value = String(folder.folder_id);
    folderBox.appendChild(folderSelect);
    row.append(weightBox, folderBox);
    card.appendChild(row);

    // the example VMSAM checks the rule on: built, or asked for
    const exampleBox = el('div', 'mt-4 rul-example-box');
    exampleBox.appendChild(el('label', 'label', 'Example file name sent to VMSAM'));
    const exampleAuto = el('div', 'font-mono text-sm rul-example-auto');
    const example = document.createElement('input');
    example.type = 'text';
    example.className = 'input font-mono pattern-input rul-example hidden';
    example.spellcheck = false;
    example.autocomplete = 'off';
    example.placeholder = 'A real file name this regex matches';
    const exampleMsg = el('div', 'mt-1 text-xs validation-msg rul-example-msg');
    exampleBox.append(exampleAuto, example, exampleMsg);
    card.appendChild(exampleBox);

    const footer = el('div', 'group-footer');
    const remove = el('button', 'btn btn-secondary btn-sm rul-delete', 'Delete');
    const save = el('button', 'btn btn-primary btn-sm rul-save', 'Save');
    remove.type = 'button';
    save.type = 'button';
    footer.append(remove, save);
    card.appendChild(footer);

    const setMsg = (node, text, tone) => { node.textContent = text; node.className = `mt-1 text-xs validation-msg ${tone}` + (node === exampleMsg ? ' rul-example-msg' : ''); };
    const values = () => {
        const renameValue = rename.value;
        const built = proposeRuleExample(rule.regex_pattern, renameValue.includes(EPISODE_PLACEHOLDER) ? renameValue : '', rule.rename_pattern);
        return {
            rename: renameValue,
            weight: Number(weight.value),
            folderId: parseInt(folderSelect.value, 10),
            built,
            example: built !== null ? built : example.value.trim()
        };
    };

    // Returns the problem that forbids saving, or null.
    const refresh = () => {
        const v = values();
        let problem = null;
        if (v.rename && !v.rename.includes(EPISODE_PLACEHOLDER)) { setMsg(renameMsg, `⚠ Must contain ${EPISODE_PLACEHOLDER}`, 'text-warning'); problem = `the rename pattern lacks ${EPISODE_PLACEHOLDER}`; }
        else setMsg(renameMsg, v.rename ? '' : 'No rename: the rule keeps the incoming name', 'text-muted');
        if (!Number.isInteger(v.weight) || v.weight < 1) problem = problem || 'the weight must be a whole number of 1 or more';
        exampleAuto.classList.toggle('hidden', v.built === null);
        example.classList.toggle('hidden', v.built !== null);
        if (v.built !== null) {
            exampleAuto.textContent = v.built;
            const episode = parseInt(extractEpisode(rule.regex_pattern, v.built), 10);
            const fromRename = [v.rename, rule.rename_pattern].some(r => r && r.includes(EPISODE_PLACEHOLDER) && ruleExampleName(r) === v.built);
            setMsg(exampleMsg, `✓ built from the ${fromRename ? 'rename pattern' : 'regex'}: episode ${episode}`, 'text-success');
        } else if (!v.example) {
            setMsg(exampleMsg, 'No name built from the rename pattern or the regex matches it: type a real file name this regex matches', 'text-warning');
            problem = problem || 'an example file name is required';
        } else if (testPattern(rule.regex_pattern, [v.example]).allValid) {
            setMsg(exampleMsg, `✓ episode ${parseInt(extractEpisode(rule.regex_pattern, v.example), 10)}`, 'text-success');
        } else {
            setMsg(exampleMsg, '⚠ The regex does not extract a valid episode number from this name', 'text-warning');
            problem = problem || 'the regex does not extract a valid episode from the example';
        }
        const dirty = v.rename !== (rule.rename_pattern || '') || v.weight !== rule.weight || v.folderId !== folder.folder_id;
        save.disabled = !!problem || !dirty;
        save.title = problem ? `Cannot save: ${problem}` : (dirty ? '' : 'Nothing changed');
        card.classList.toggle('rul-dirty', dirty);
        return problem;
    };
    rename.oninput = refresh;
    weight.oninput = refresh;
    folderSelect.onchange = refresh;
    example.oninput = refresh;
    save.onclick = () => saveRuleEdit(rule, folder, folders, values(), refresh, save);
    remove.onclick = () => deleteRuleFromList(rule, remove);
    refresh();
    return card;
}

async function saveRuleEdit(rule, folder, folders, v, refresh, button) {
    const problem = refresh();
    if (problem) { showToast(`Cannot save: ${problem}`, 'error'); return; }
    const target = folders.find(f => f.folder_id === v.folderId);
    if (!target) { showToast(`Folder ${v.folderId} is not in the folder list`, 'error'); return; }
    if (v.folderId !== folder.folder_id) {
        const confirmed = await confirmDialog(`${rule.regex_pattern}\n\nfrom ${folder.destination_path}\nto   ${target.destination_path}\n\nNew downloads this rule matches will go to the new folder; files already registered stay where they are.`, { title: 'Move this rule to another folder?', okText: 'Move rule' });
        if (!confirmed) return;
    }
    const payload = {
        regex_pattern: rule.regex_pattern,
        rename_pattern: v.rename ? v.rename : null,
        weight: v.weight,
        destination_path: target.destination_path,
        example_filename: v.example
    };
    button.disabled = true;
    try {
        const result = await api.saveRegex(payload);
        const moved = v.folderId !== folder.folder_id;
        showToast(`${result.message || 'Rule updated'}` + (moved ? `, moved to ${target.destination_path}` : ''), 'success', 5000);
    } catch (e) {
        showToast('Rule not saved: ' + e.message, 'error', 8000);
        button.disabled = false;
        const msg = button.closest('.rul-card').querySelector('.rul-example-msg');
        msg.textContent = `VMSAM refused: ${e.message}`;
        msg.className = 'mt-1 text-xs validation-msg text-danger rul-example-msg';
        return;
    }
    await loadRulesList();
}

// "Edit regex in Bulk Regex": that tab, on this folder, with the rule on a
// replace card holding the real files it matches.
async function openRegexEdit(folder, rule) {
    switchTab('regex-tab');
    if (state.selectedFolderId !== folder.folder_id) regexTab.picker.select(folder);
    await regexTab.rulesReady;
    await startRegexReplace(rule);
}

async function deleteRuleFromList(rule, button) {
    const confirmed = await confirmDialog(`${rule.regex_pattern}\n→ ${rule.rename_pattern || '(no rename)'}\n\nFiles already registered stay as they are; new downloads only this rule matched will no longer be integrated.`, { title: 'Delete this rule?', okText: 'Delete' });
    if (!confirmed) return;
    button.disabled = true;
    try {
        const result = await api.deleteRegex(rule.regex_pattern);
        showToast(result.message || 'Rule deleted', 'success');
    } catch (e) {
        showToast('Rule not deleted: ' + e.message, 'error', 8000);
        button.disabled = false;
        return;
    }
    await loadRulesList();
}

// --- 8g. Errors tab ---
//
// The files the pipeline rejected (GET /errors), grouped by folder with the
// registered episode they failed to merge with (the master). A rejected file
// can be queued for a new fusion (POST /fusion) or deleted (POST
// /errors/delete/: the file, its .log.error, its row, and its error directory
// when only .error files remain). The fusion queue (GET /fusion) is polled
// every 5 s while the tab is shown. Each folder group folds (its header is a
// button); "Expand all" / "Collapse all"; the search box keeps the folders
// whose path, or one of whose file names, contains the text (case-insensitive;
// a folder matched by file names shows those files only). On arrival
// everything is listed and unfolded; a reload after a delete keeps both.

let errorsTab = null;
const FUSION_POLL_MS = 5000;

function initErrorsTab() {
    if (!errorsTab) {
        errorsTab = { timer: null, queue: null, collapsed: new Set() };
        document.getElementById('errors-search').addEventListener('input', applyErrorsSearch);
        document.getElementById('errors-expand-all').onclick = () => setAllErrorGroups(false);
        document.getElementById('errors-collapse-all').onclick = () => setAllErrorGroups(true);
    }
    // the overview first: everything listed and unfolded
    errorsTab.collapsed = new Set();
    document.getElementById('errors-search').value = '';
    loadErrorsList();
    refreshFusionQueue();
    startFusionPolling();
}

function errorsTabVisible() {
    const section = document.getElementById('errors-tab');
    return !!section && !section.classList.contains('hidden');
}

// One timer, stopped by itself once the tab is hidden; initErrorsTab starts it again.
function startFusionPolling() {
    if (errorsTab.timer) return;
    errorsTab.timer = setInterval(() => {
        if (!errorsTabVisible()) {
            clearInterval(errorsTab.timer);
            errorsTab.timer = null;
            return;
        }
        if (!document.hidden) refreshFusionQueue();
    }, FUSION_POLL_MS);
}

async function refreshFusionQueue() {
    try {
        errorsTab.queue = await api.getFusionStatus();
        renderFusionQueue(errorsTab.queue, null);
    } catch (e) {
        errorsTab.queue = null;
        renderFusionQueue(null, e.message);
    }
    markQueuedErrors();
}

function renderFusionQueue(queue, error) {
    const box = document.getElementById('errors-queue');
    box.innerHTML = '';
    const time = new Date().toLocaleTimeString();
    if (error) {
        box.appendChild(el('div', 'text-danger err-queue-error', `Queue unavailable: ${error}`));
        box.appendChild(el('div', 'text-muted text-xs', `checked at ${time}`));
        return;
    }
    const status = el('div', 'err-queue-line');
    status.appendChild(el('span', 'text-muted', 'Status: '));
    status.appendChild(el('span', `badge ${queue.status === 'idle' ? '' : 'badge-success'} err-queue-status`, queue.status || 'unknown'));
    status.appendChild(el('span', 'text-muted err-queue-gap', 'Waiting: '));
    status.appendChild(el('span', 'font-mono err-queue-length', String(queue.queue_length !== undefined ? queue.queue_length : (queue.pending_jobs || []).length)));
    box.appendChild(status);
    const current = el('div', 'err-queue-line');
    current.appendChild(el('span', 'text-muted', 'Current job: '));
    if (queue.current_job && queue.current_job.error_file_path) {
        const name = el('span', 'font-mono text-accent err-queue-current', baseName(queue.current_job.error_file_path));
        name.title = queue.current_job.error_file_path;
        current.appendChild(name);
        if (queue.current_job.started_at) current.appendChild(el('span', 'text-muted text-xs', ` (started ${queue.current_job.started_at})`));
    } else {
        current.appendChild(el('span', 'text-muted err-queue-current', 'none'));
    }
    box.appendChild(current);
    const pending = queue.pending_jobs || [];
    if (pending.length) {
        box.appendChild(el('div', 'text-muted', 'Pending jobs:'));
        const list = el('ol', 'font-mono err-queue-pending');
        pending.forEach(job => {
            const item = el('li', null, baseName(job.error_file_path));
            item.title = job.error_file_path;
            list.appendChild(item);
        });
        box.appendChild(list);
    } else {
        box.appendChild(el('div', 'text-muted err-queue-pending-none', 'No job waiting.'));
    }
    box.appendChild(el('div', 'text-muted text-xs', `refreshed at ${time}`));
}

// Badges "processing" / "queued" on the rows the queue holds.
function markQueuedErrors() {
    const queue = errorsTab && errorsTab.queue;
    const current = queue && queue.current_job ? queue.current_job.error_file_path : null;
    const pending = new Set(queue ? (queue.pending_jobs || []).map(job => job.error_file_path) : []);
    document.querySelectorAll('#errors-groups tr[data-path]').forEach(row => {
        const badge = row.querySelector('.err-queue-badge');
        const path = row.dataset.path;
        const text = path === current ? 'processing' : pending.has(path) ? 'queued' : '';
        badge.textContent = text;
        badge.classList.toggle('hidden', !text);
    });
}

async function loadErrorsList() {
    const container = document.getElementById('errors-groups');
    const count = document.getElementById('errors-count');
    count.textContent = 'Loading…';
    let files = [];
    try {
        files = ((await api.getErrors()) || {}).incompatible_files || [];
    } catch (e) {
        count.textContent = '';
        container.innerHTML = '';
        container.appendChild(el('div', 'text-danger text-sm', `Failed to load: ${e.message}`));
        return;
    }
    folderCache = null;
    const folders = await fetchFolders().catch(() => []);
    const folderById = new Map(folders.map(f => [f.folder_id, f]));
    const groups = new Map();
    files.forEach(file => {
        if (!groups.has(file.folder_id)) groups.set(file.folder_id, []);
        groups.get(file.folder_id).push(file);
    });
    const ids = Array.from(groups.keys()).sort((a, b) => {
        const pa = folderById.has(a) ? folderById.get(a).destination_path : `~${a}`;
        const pb = folderById.has(b) ? folderById.get(b).destination_path : `~${b}`;
        return pa.localeCompare(pb);
    });
    // master episodes per folder: Map(episode_number -> episode), or an error text
    const masters = new Map();
    await Promise.all(ids.map(async id => {
        try {
            const response = await api.getFolderEpisodes(id);
            masters.set(id, new Map(((response && response.episodes) || []).map(ep => [ep.episode_number, ep])));
        } catch (e) {
            masters.set(id, e.message);
        }
    }));

    container.innerHTML = '';
    count.textContent = `${files.length} rejected file${files.length === 1 ? '' : 's'}` + (ids.length ? ` in ${ids.length} folder${ids.length === 1 ? '' : 's'}` : '');
    if (!files.length) container.appendChild(el('div', 'text-muted text-sm', 'No rejected file.'));
    ids.forEach(id => {
        const folder = folderById.get(id);
        const folderFiles = groups.get(id);
        const group = el('div', 'err-group');
        group.dataset.folderId = String(id);
        const path = folder ? folder.destination_path : `Folder ${id} (not in the folder list)`;
        group.dataset.path = path;
        const head = el('button', 'err-group-head');
        head.type = 'button';
        head.appendChild(el('span', 'err-group-icon', '▼'));
        head.appendChild(el('span', 'font-mono err-group-title', path));
        head.appendChild(el('span', 'badge err-group-count', `${folderFiles.length} file${folderFiles.length === 1 ? '' : 's'}`));
        head.onclick = () => setErrorGroupCollapsed(group, !group.classList.contains('collapsed'));
        group.appendChild(head);
        const body = el('div', 'err-group-body');
        body.id = `err-group-body-${id}`;
        head.setAttribute('aria-controls', body.id);
        group.appendChild(body);
        const folderMasters = masters.get(id);
        if (typeof folderMasters === 'string') body.appendChild(el('div', 'text-warning text-xs', `Master episodes unavailable: ${folderMasters}`));
        const table = el('table', 'report-table err-table');
        const headRow = el('tr');
        ['Ep', 'Rejected file', 'Weight', 'Master episode file', '', ''].forEach(h => headRow.appendChild(el('th', null, h)));
        const thead = el('thead');
        thead.appendChild(headRow);
        const tbody = el('tbody');
        folderFiles.slice().sort((a, b) => a.episode_number - b.episode_number || a.file_path.localeCompare(b.file_path)).forEach(file => {
            tbody.appendChild(renderErrorRow(file, folderMasters instanceof Map ? folderMasters.get(file.episode_number) : null));
        });
        table.append(thead, tbody);
        const wrap = el('div', 'err-table-wrap');
        wrap.appendChild(table);
        body.appendChild(wrap);
        setErrorGroupCollapsed(group, errorsTab.collapsed.has(String(id)));
        container.appendChild(group);
    });
    applyErrorsSearch();
    markQueuedErrors();
}

function setErrorGroupCollapsed(group, collapsed) {
    group.classList.toggle('collapsed', collapsed);
    group.querySelector('.err-group-body').classList.toggle('hidden', collapsed);
    group.querySelector('.err-group-icon').textContent = collapsed ? '▶' : '▼';
    group.querySelector('.err-group-head').setAttribute('aria-expanded', String(!collapsed));
    if (collapsed) errorsTab.collapsed.add(group.dataset.folderId);
    else errorsTab.collapsed.delete(group.dataset.folderId);
}

// Expand all / Collapse all: the folders the search shows.
function setAllErrorGroups(collapsed) {
    document.querySelectorAll('#errors-groups .err-group:not(.hidden)').forEach(group => setErrorGroupCollapsed(group, collapsed));
}

function applyErrorsSearch() {
    const query = document.getElementById('errors-search').value.trim().toLowerCase();
    const groups = Array.from(document.querySelectorAll('#errors-groups .err-group'));
    let shown = 0;
    groups.forEach(group => {
        const pathMatch = !query || group.dataset.path.toLowerCase().includes(query);
        let rowsShown = 0;
        group.querySelectorAll('tr[data-path]').forEach(row => {
            const visible = pathMatch || baseName(row.dataset.path).toLowerCase().includes(query);
            row.classList.toggle('hidden', !visible);
            if (visible) rowsShown++;
        });
        const visible = pathMatch || rowsShown > 0;
        group.classList.toggle('hidden', !visible);
        if (visible) shown++;
    });
    const note = document.getElementById('errors-shown');
    note.textContent = query && groups.length ? `${shown} of ${groups.length} folder${groups.length === 1 ? '' : 's'} shown` : '';
    let none = document.querySelector('#errors-groups .err-search-none');
    if (query && groups.length && !shown) {
        if (!none) {
            none = el('div', 'text-muted text-sm err-search-none', 'No folder or file name contains this text.');
            document.getElementById('errors-groups').appendChild(none);
        }
    } else if (none) none.remove();
}

function renderErrorRow(file, master) {
    const row = el('tr', 'err-row');
    row.dataset.path = file.file_path;
    const name = el('td', 'font-mono err-file', baseName(file.file_path));
    name.title = file.file_path;
    const masterCell = el('td', 'font-mono err-master');
    if (master) {
        masterCell.textContent = `${baseName(master.file_path)} (W ${master.file_weight})`;
        masterCell.title = master.file_path;
    } else {
        masterCell.textContent = 'no episode registered';
        masterCell.classList.add('text-warning');
    }
    const status = el('td', 'err-status');
    status.appendChild(el('span', 'badge badge-success err-queue-badge hidden'));
    const message = el('div', 'text-xs err-msg');
    status.appendChild(message);
    const actions = el('td', 'err-actions');
    const relaunch = el('button', 'btn btn-primary btn-sm err-relaunch', 'Relaunch fusion');
    const remove = el('button', 'btn btn-secondary btn-sm err-delete', 'Delete');
    relaunch.type = 'button';
    remove.type = 'button';
    relaunch.onclick = () => relaunchFusion(file, relaunch, message);
    remove.onclick = () => deleteErrorFile(file, remove, message);
    actions.append(relaunch, remove);
    row.append(el('td', 'font-mono', padEpisode(file.episode_number)), name,
        el('td', 'font-mono', file.file_weight === null || file.file_weight === undefined ? '-' : String(file.file_weight)),
        masterCell, status, actions);
    return row;
}

async function relaunchFusion(file, button, message) {
    button.disabled = true;
    message.textContent = '';
    try {
        const result = await api.queueFusion(file.file_path);
        const text = (result.message || 'Fusion job queued') + (result.queue_position ? ` (position ${result.queue_position})` : '');
        message.textContent = text;
        message.className = 'text-xs err-msg text-success';
        showToast(text, 'success');
    } catch (e) {
        message.textContent = e.message;
        message.className = 'text-xs err-msg text-danger';
        showToast('Fusion not queued: ' + e.message, 'error', 8000);
    } finally {
        button.disabled = false;
    }
    refreshFusionQueue();
}

async function deleteErrorFile(file, button, message) {
    const confirmed = await confirmDialog(`${baseName(file.file_path)}\n\nDeletes this rejected file, its .log.error file and its row. Its error directory is removed as well when only .error files remain in it.\nThe registered episode is not touched. This cannot be undone.`, { title: 'Delete this rejected file?', okText: 'Delete' });
    if (!confirmed) return;
    button.disabled = true;
    let result;
    try {
        result = await api.deleteError(file.file_path);
    } catch (e) {
        message.textContent = e.message;
        message.className = 'text-xs err-msg text-danger';
        showToast('Not deleted: ' + e.message, 'error', 8000);
        button.disabled = false;
        return;
    }
    const removed = result.removed_directories || [];
    showToast((result.message || 'Deleted') + (removed.length ? `: ${removed.length} director${removed.length === 1 ? 'y' : 'ies'} removed` : ''), 'success', 5000);
    showErrorsReport(result);
    await loadErrorsList();
}

// What the last delete did, kept above the list.
function showErrorsReport(result) {
    const box = document.getElementById('errors-report');
    box.innerHTML = '';
    box.classList.remove('hidden');
    box.appendChild(el('div', 'text-success', `${result.message || 'Deleted'}: ${baseName(result.error_file_path)}` + (result.file_was_missing ? ' (the file was already missing on disk)' : '')));
    const section = (label, items, className) => {
        if (!items || !items.length) return;
        box.appendChild(el('div', 'text-muted mt-2', label));
        const list = el('ul', `font-mono err-report-list ${className}`);
        items.forEach(item => list.appendChild(el('li', null, item)));
        box.appendChild(list);
    };
    section('Deleted files:', result.deleted_files, 'err-report-files');
    section('Removed directories:', result.removed_directories, 'err-report-dirs');
    section('Warnings:', result.warnings, 'err-report-warnings text-warning');
    if (!(result.removed_directories || []).length) box.appendChild(el('div', 'text-muted mt-2', 'No directory removed (files other than .error ones are still there).'));
}

// Exposed for the engine self-test page (no module system, plain script).
window.vmsamEngine = { tokenizeName, buildTemplate, analyzeGroup, testPattern, groupByPrefix, groupByNamePattern, groupByNameShape, refineGroupsByPairs, nameShape, findMarkerSpan, extractEpisode, buildRegex };
switchSection(currentSection);
