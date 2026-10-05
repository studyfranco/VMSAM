// VMSAM web interface — vanilla JS, no external dependency.
//
// Sections:
//   1. utilities (toast, toggles)
//   2. API client
//   3. filename analysis engine (pure functions, no DOM)
//   4. tabs
//   5. folder creation
//   6. shared widgets: folder picker, existing-rules panel, file modal
//   7. group board (regex / index / incrementaller cards, drag & drop)
//   8. tabs: bulk regex, index folder, specials, incrementaller

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
    getFolders() { return apiFetch('/api/vmsam/folders_list'); },
    createFolder(payload) { return postJson('/api/vmsam/folders', payload); },
    createRegex(payload) { return postJson('/api/vmsam/regex', payload); },
    getRegexes(folderId) { return apiFetch(`/api/vmsam/regex_list?folder_id=${folderId}`); },
    indexFolder(folderId) { return postJson('/api/vmsam/index_folder', { folder_id: folderId }); },
    createSpecial(payload) { return postJson('/api/vmsam/special', payload); },
    getSpecials() { return apiFetch('/api/vmsam/special_list'); },
    createIncrementaller(payload) { return postJson('/api/vmsam/incrementaller', payload); },
    getIncrementallers() { return apiFetch('/api/vmsam/incrementaller_list'); },
    getFolderEpisodes(folderId) { return apiFetch(`/api/vmsam/episodes_folder?folder_id=${folderId}`); },
    // newPath is relative to CREATE_ROOT: the relay prefixes the root
    moveFolder(folderId, newPath) { return postJson('/api/vmsam/folders_move', { folder_id: folderId, new_destination_path: newPath }); }
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
// of an SxxE marker every name carries, else the best score. The proposed
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
const MATCH_WEIGHT_DIGITS = 25;
// Each run of unmatched tokens costs a little, far less than any match: among
// alignments of equal weight, the one with the fewest holes wins, so the dots
// around a changing title stay with the words they separate.
const GAP_OPEN_COST = 1;

function matchWeight(a, b) {
    if (sameToken(a, b)) return MATCH_WEIGHT_SAME;
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

// Index of the episode digit run of an S<season>E<episode> marker, or -1.
// `S` must be its own letter run (`.S01E03.`, `s02e05`, ` S1E3 `), or the
// capital end of a camel-cased word (`TitleS01E03`); `E` must stand alone.
function findEpisodeMarker(tokens) {
    for (let k = 0; k + 3 < tokens.length; k++) {
        const s = tokens[k].v;
        const isMarkerS = /^s$/i.test(s) || /[a-zà-ÿ]S$/.test(s);
        if (isMarkerS && tokens[k + 1].d && /^e$/i.test(tokens[k + 2].v) && tokens[k + 3].d) return k + 3;
    }
    return -1;
}

// A short literal run (at most three tokens) squeezed between two wild spans
// is a coincidence of the titles (`.Is.` in two different titles): make it
// wild too, so the regex holds one `.+` and the rename drops it. A changing
// digit run glued after a wild word is the end of that word (`ABCD1234`, a
// CRC): wild too, unless that word is an episode prefix (`E`, `Ep`). The SxxE
// marker is never absorbed.
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
//   markerIndex: the episode digits of an SxxE marker every name carries,
//                aligned in every name (null otherwise)
function buildTemplate(names) {
    const ref = tokenizeName(names[0]);
    const slots = ref.map(t => ({ v: t.v, d: t.d, start: t.start, end: t.start + t.v.length, varying: false, wild: false, optional: false, extraBefore: false }));
    let extraAfterEnd = false;
    const refMarker = findEpisodeMarker(ref);
    let markerAligned = refMarker >= 0;

    for (const name of names.slice(1)) {
        const other = tokenizeName(name);
        const pairs = lcsPairs(ref, other);
        if (markerAligned) {
            const otherMarker = findEpisodeMarker(other);
            markerAligned = otherMarker >= 0 && pairs.some(p => p[0] === refMarker && p[1] === otherMarker);
        }
        let prev = [-1, -1];
        for (const pair of [...pairs, [ref.length, other.length]]) {
            const [i2, j2] = pair;
            const [i1, j1] = prev;
            if (i2 < ref.length && j2 < other.length && !sameToken(ref[i2], other[j2])) slots[i2].varying = true;
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
    absorbBetweenWilds(slots, markerIndex === null ? -1 : markerIndex - 3, markerIndex === null ? -1 : markerIndex);
    return { name: names[0], slots, extraAfterEnd, count: names.length, markerIndex };
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

    // An SxxE marker in every name of the group wins outright.
    if (template.markerIndex === index) add(1000, 'SxxEyy in every name');
    else if (/S\d{1,3}[ ._-]?E$/i.test(before)) add(100, 'SxxEyy');
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
function buildRename(template, episodeIndex) {
    if (episodeIndex === null || episodeIndex === undefined) return template.name;
    const GAP = '\u0000';
    let out = '';
    template.slots.forEach((slot, i) => {
        if (slot.wild) { if (!out.endsWith(GAP)) out += GAP; return; }
        out += i === episodeIndex ? EPISODE_PLACEHOLDER : slot.v;
    });
    if (!out.includes(GAP)) return out;
    // A bracket pair that only held a dropped span goes with it.
    let previous;
    do {
        previous = out;
        out = out.replace(/[\[(][\s._\-]*\u0000[\s._\-]*[\])]/g, GAP).replace(/\u0000+/g, GAP);
    } while (out !== previous);
    out = out.replace(/([\s._\-]*)\u0000([\s._\-]*)/g, (match, left, right, offset, whole) => {
        const before = whole.slice(0, offset);
        const after = whole.slice(offset + match.length);
        if (!before || !after) return '';
        if (right.endsWith('.') && /^[A-Za-z0-9]{1,5}$/.test(after)) return '.';
        return left || right;
    });
    return out;
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

// The whole proposal for one group.
function analyzeGroup(names, forcedEpisodeIndex) {
    const template = buildTemplate(names);
    const pick = pickEpisodeSlot(template);
    let episodeIndex = pick.best;
    if (forcedEpisodeIndex !== null && forcedEpisodeIndex !== undefined && numberSlots(template).includes(forcedEpisodeIndex)) {
        episodeIndex = forcedEpisodeIndex;
    }
    const regex = buildRegex(template, episodeIndex);
    return {
        template,
        episodeIndex,
        candidates: pick.candidates,
        regex,
        rename: buildRename(template, episodeIndex),
        check: testPattern(regex, names)
    };
}

// Default grouping for a folder being indexed: same first five characters.
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
    'special-tab': () => initSpecialTab(),
    'incr-tab': () => initIncrementallerTab()
};

function switchTab(tabId) {
    document.querySelectorAll('main > section').forEach(el => el.classList.add('hidden'));
    document.getElementById(tabId).classList.remove('hidden');
    document.querySelectorAll('.nav-group button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    if (TAB_INIT[tabId]) TAB_INIT[tabId]();
}

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
        const result = await api.createFolder(payload);
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

function renderRuleRow(rule) {
    const row = el('div', 'rule-row');
    const left = el('div');
    left.appendChild(el('div', 'font-mono text-accent mb-1', rule.regex_pattern));
    left.appendChild(el('div', 'text-muted text-sm', `→ ${rule.rename_pattern || '(no rename)'}`));
    const right = el('div', 'badge', rule.weight !== undefined ? `W: ${rule.weight}` : `+${rule.episode_incremental}`);
    row.append(left, right);
    return row;
}

// Collapsible "Existing rules" panel. Returns the rules so the caller can
// test files against them.
async function loadExistingRulesInto(contentId, loader, emptyText) {
    const content = document.getElementById(contentId);
    content.innerHTML = '';
    content.appendChild(el('div', 'text-sm text-muted', 'Loading...'));
    try {
        const rules = await loader();
        content.innerHTML = '';
        if (!rules.length) {
            content.appendChild(el('div', 'text-sm text-muted', emptyText));
        } else {
            rules.forEach(rule => content.appendChild(renderRuleRow(rule)));
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

    try {
        const files = await api.listFiles('', 'files');
        const vidFiles = files.filter(f => !f.is_dir && isVideoName(f.name));
        list.innerHTML = '';
        if (!vidFiles.length) list.appendChild(el('div', 'text-muted p-4', 'No video file in the download root'));
        vidFiles.forEach(f => {
            const row = el('div', 'file-row');
            const check = document.createElement('input');
            check.type = 'checkbox';
            check.value = f.name;
            check.className = 'input-checkbox';
            row.append(check, el('span', 'text-main', f.name));
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
        };
    } catch (e) {
        list.innerHTML = '';
        list.appendChild(el('div', 'text-accent p-4', e.message));
    }
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

    board.addFiles = (names, grouped) => {
        const known = new Set(board.groups.flatMap(g => g.files));
        const fresh = names.filter(n => !known.has(n));
        if (!fresh.length) return;
        board.addGroups(grouped ? groupByPrefix(fresh) : fresh.map(n => [n]));
    };

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
        board.groups = board.groups.filter(g => g.files.length);
        board.render();
    };

    board.removeGroup = (id) => {
        board.groups = board.groups.filter(g => g.id !== id);
        board.render();
    };

    board.proposal = (group) => analyzeGroup(group.files, group.episodeIndex);

    board.values = (group) => {
        const proposal = board.proposal(group);
        return {
            regex: group.regexManual !== null ? group.regexManual : proposal.regex,
            rename: group.renameManual !== null ? group.renameManual : proposal.rename,
            proposal
        };
    };

    board.render = () => {
        container.innerHTML = '';
        if (!board.groups.length) {
            container.appendChild(el('div', 'text-muted text-sm empty-board', options.emptyText || 'No file yet.'));
            return;
        }
        board.groups.forEach(group => container.appendChild(renderGroupCard(board, group)));
        const zone = el('div', 'drop-zone', 'Drop a file here to start a new group');
        wireDropTarget(zone, board, null);
        container.appendChild(zone);
    };

    return board;
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
    const check = testPattern(regex, group.files);
    const card = el('div', `card group-card${group.saved ? ' saved' : ''}`);
    wireDropTarget(card, board, group.id);

    // Header
    const header = el('div', 'group-header');
    const title = el('div');
    title.appendChild(el('span', 'text-muted text-xs uppercase font-bold', board.mode === 'incrementaller' ? 'Increment rule' : 'Rule'));
    title.appendChild(el('span', 'badge', `${group.files.length} file${group.files.length > 1 ? 's' : ''}`));
    if (group.saved) title.appendChild(el('span', 'badge badge-success', 'saved ✓'));
    const remove = el('button', 'btn-icon', '🗑️');
    remove.title = 'Remove this group';
    remove.onclick = () => board.removeGroup(group.id);
    header.append(title, remove);
    card.appendChild(header);

    // Files
    const list = el('div', 'group-files');
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
        if (group.files.length > 1) {
            const eject = el('button', 'btn-icon eject', '−');
            eject.title = 'Move this file to its own group';
            eject.onclick = () => board.moveFile(name, group.id, null);
            row.appendChild(eject);
        }
        list.appendChild(row);
    });
    card.appendChild(list);

    // Reference name with clickable digit runs
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
            chip.onclick = () => { group.episodeIndex = i; group.regexManual = null; group.renameManual = null; group.saved = false; board.render(); };
            line.appendChild(chip);
        } else {
            line.appendChild(el('span', slot.wild ? 'wild-part' : (slot.varying ? 'varying-part' : null), slot.v));
        }
    });
    example.appendChild(line);
    card.appendChild(example);

    // Inputs
    const grid = el('div', 'grid-2');
    const regexBox = el('div');
    regexBox.appendChild(el('label', 'label', 'Regex Pattern (Python)'));
    const regexInput = document.createElement('input');
    regexInput.type = 'text';
    regexInput.className = 'input font-mono';
    regexInput.value = regex;
    regexInput.oninput = () => { group.regexManual = regexInput.value; group.saved = false; refreshCheck(); };
    regexBox.appendChild(regexInput);
    const regexMsg = el('div', 'mt-2 text-xs validation-msg');
    regexBox.appendChild(regexMsg);

    const renameBox = el('div');
    renameBox.appendChild(el('label', 'label', 'Rename Pattern'));
    const renameInput = document.createElement('input');
    renameInput.type = 'text';
    renameInput.className = 'input font-mono';
    renameInput.value = rename;
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
        numberInput.oninput = () => { group.weight = parseInt(numberInput.value, 10) || 1; group.saved = false; };
    }
    numberBox.appendChild(numberInput);
    const summary = el('div', 'extraction-box');
    row.append(numberBox, summary);
    card.appendChild(row);

    // Footer
    const footer = el('div', 'group-footer');
    const repropose = el('button', 'btn btn-secondary btn-sm', 'Re-propose');
    repropose.onclick = () => { group.regexManual = null; group.renameManual = null; group.saved = false; board.render(); };
    const save = el('button', 'btn btn-primary btn-sm', board.mode === 'incrementaller' ? 'Save Increment Rule' : 'Save Rule');
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
        } else if (result.allValid) {
            regexMsg.textContent = '✓ Every file yields an episode number';
            regexMsg.className = 'mt-2 text-xs text-success validation-msg';
        } else {
            const failing = result.perFile.filter(f => !f.valid).length;
            regexMsg.textContent = `⚠ ${failing} file${failing > 1 ? 's' : ''} without a valid 'episode' group`;
            regexMsg.className = 'mt-2 text-xs text-warning validation-msg';
        }
        if (!currentRename.includes(EPISODE_PLACEHOLDER)) {
            renameMsg.textContent = `⚠ Must contain ${EPISODE_PLACEHOLDER}`;
            renameMsg.className = 'mt-2 text-xs text-warning validation-msg';
        } else {
            renameMsg.textContent = '';
            renameMsg.className = 'mt-2 text-xs text-muted validation-msg';
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

async function saveGroup(board, group, button) {
    const { regex, rename } = board.values(group);
    if (!regex) { showToast('Regex pattern is required', 'error'); return; }
    if (!rename.includes(EPISODE_PLACEHOLDER)) { showToast(`Rename pattern must contain ${EPISODE_PLACEHOLDER}`, 'error'); return; }
    const check = testPattern(regex, group.files);
    if (!check.allValid) { showToast('Every file of the group must yield a valid episode number', 'error'); return; }

    let payload;
    if (board.mode === 'incrementaller') {
        payload = { regex_pattern: regex, rename_pattern: rename, episode_incremental: group.increment, example_filename: group.files[0] };
    } else {
        const folderPath = board.getFolderPath();
        if (!folderPath) { showToast('Select a folder first', 'error'); return; }
        payload = { regex_pattern: regex, rename_pattern: rename, weight: group.weight, example_filename: group.files[0], destination_path: folderPath };
    }

    button.disabled = true;
    try {
        const result = board.mode === 'incrementaller' ? await api.createIncrementaller(payload) : await api.createRegex(payload);
        group.saved = true;
        const extra = result && result.new_file_name ? ` → ${result.new_file_name}` : '';
        showToast((result && result.message ? result.message : 'Rule saved') + extra, 'success');
        board.render();
        if (board.onSaved) board.onSaved(group, result);
    } catch (e) {
        showToast('Error saving rule: ' + e.message, 'error', 6000);
    } finally {
        button.disabled = false;
    }
}

// --- 8a. Bulk Regex tab ---

let regexTab = null;

function initRegexTab() {
    if (regexTab) { regexTab.picker.load(); return; }
    const board = createGroupBoard({
        containerId: 'regex-cards', mode: 'regex',
        getFolderPath: () => state.selectedFolderPath,
        emptyText: 'Add files: one rule per file by default, drag a file onto another rule to group them.'
    });
    const picker = createFolderPicker('regex', (f) => {
        state.selectedFolderId = f.id;
        state.selectedFolderPath = f.destination_path;
        document.getElementById('regex-work-area').classList.remove('disabled-area');
        document.getElementById('regex-existing').classList.remove('hidden');
        loadExistingRulesInto('regex-existing-content', () => loadFolderRules(f.id), 'No existing rules for this folder.');
    }, () => {
        state.selectedFolderId = null;
        state.selectedFolderPath = null;
        document.getElementById('regex-work-area').classList.add('disabled-area');
        document.getElementById('regex-existing').classList.add('hidden');
        board.clear();
    });
    board.onSaved = () => loadExistingRulesInto('regex-existing-content', () => loadFolderRules(state.selectedFolderId), 'No existing rules for this folder.');
    regexTab = { board, picker };
    picker.load();
    board.render();
}

function clearFolderSelection() {
    if (regexTab) regexTab.picker.clear();
}

function openRegexFileModal() {
    openFileModal((names) => regexTab.board.addFiles(names, false));
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
        document.getElementById('index-work-area').classList.remove('disabled-area');
        document.getElementById('index-report').classList.add('hidden');
        showIndexMovePanel(f);
        loadIndexFolderFiles();
    }, () => {
        indexTab.folder = null;
        document.getElementById('index-work-area').classList.add('disabled-area');
        document.getElementById('index-move').classList.add('hidden');
        board.clear();
    });
    // A save refreshes the rule and covered lists only: rebuilding the board
    // would throw away the edits made on the other groups.
    board.onSaved = () => loadIndexFolderFiles(false);
    indexTab = { board, picker, folder: null, rules: [], registerPath: null, registerDir: '' };
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
    const folder = folders.find(f => f.id === id);
    if (folder) indexTab.picker.select(folder);
    else showToast(`Folder ${id} is not in the folder list`, 'error');
}

// --- Register an existing folder (a directory of CREATE_ROOT VMSAM does not know) ---

function toggleIndexRegister() {
    const content = document.getElementById('index-register-content');
    const opening = content.classList.contains('hidden');
    toggleCollapsible('index-register-content', 'index-register-icon');
    if (opening) loadIndexRegisterBrowser(indexTab ? indexTab.registerDir : '');
}

async function loadIndexRegisterBrowser(path = '') {
    const container = document.getElementById('index-register-browser');
    container.innerHTML = '';
    container.appendChild(el('div', 'text-muted text-center p-4', 'Loading...'));
    indexTab.registerDir = path;
    try {
        const [items, config, folders] = await Promise.all([
            api.listFiles(path, 'create'),
            fetchUiConfig(),
            fetchFolders()
        ]);
        const registered = new Set(folders.map(f => normalizePath(f.destination_path)));
        container.innerHTML = '';

        const where = el('div', 'dir-item register-where');
        where.appendChild(el('span', 'font-mono text-muted', path ? `/${path}` : '/ (library root)'));
        container.appendChild(where);
        if (path) {
            const back = el('div', 'dir-item text-blue');
            back.appendChild(el('span', null, '📁 ..'));
            back.onclick = () => loadIndexRegisterBrowser(path.split('/').slice(0, -1).join('/'));
            container.appendChild(back);
        }

        const dirs = items.filter(i => i.is_dir);
        if (!dirs.length) container.appendChild(el('div', 'text-muted text-center p-2', 'No sub-folder here'));
        dirs.forEach(item => {
            const isRegistered = registered.has(normalizePath(`${config.create_root}/${item.path}`));
            const row = el('div', 'dir-item' + (indexTab.registerPath === item.path ? ' selected' : '') + (isRegistered ? ' dir-registered' : ''));
            row.dataset.path = item.path;
            row.onclick = () => selectIndexRegisterPath(item.path);
            row.ondblclick = () => loadIndexRegisterBrowser(item.path);

            const name = el('div', 'dir-name flex-1');
            name.appendChild(el('span', null, `📁 ${item.name}`));
            if (isRegistered) name.appendChild(el('span', 'badge badge-success register-badge', 'registered'));
            const open = el('button', 'btn btn-sm btn-secondary', 'Open →');
            open.onclick = (e) => { e.stopPropagation(); loadIndexRegisterBrowser(item.path); };
            const actions = el('div', 'dir-actions');
            actions.appendChild(open);
            row.append(name, actions);
            container.appendChild(row);
        });
    } catch (e) {
        container.innerHTML = '';
        const err = el('div', 'text-danger text-sm p-4 text-center', `Error: ${e.message}`);
        if (path) { // e.g. the browsed directory was just moved away
            const root = el('button', 'btn btn-sm btn-secondary mt-2', 'Back to the library root');
            root.onclick = () => loadIndexRegisterBrowser('');
            err.appendChild(document.createElement('br'));
            err.appendChild(root);
        }
        container.appendChild(err);
    }
}

function selectIndexRegisterPath(path) {
    indexTab.registerPath = path;
    document.getElementById('index-register-path').textContent = path;
    document.getElementById('index-register-btn').disabled = false;
    document.querySelectorAll('#index-register-browser .dir-item').forEach(row => {
        row.classList.toggle('selected', row.dataset.path === path);
    });
}

async function registerIndexFolder() {
    const path = indexTab && indexTab.registerPath;
    if (!path) { showToast('Click a folder in the browser first', 'error'); return; }
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
        const result = await api.createFolder(payload);
        showToast(result && result.message ? result.message : 'Folder registered', 'success');
        await selectIndexFolderById(result.folder_id);
        loadIndexRegisterBrowser(indexTab.registerDir); // refresh the "registered" marks
    } catch (e) {
        showToast('Cannot register the folder: ' + e.message, 'error', 6000);
    } finally {
        button.disabled = false;
    }
}

// --- Move / rename the selected folder ---

async function showIndexMovePanel(folder) {
    const panel = document.getElementById('index-move');
    const input = document.getElementById('index-move-path');
    const note = document.getElementById('index-move-note');
    panel.classList.remove('hidden');
    note.textContent = '';
    input.value = '';
    try {
        const config = await fetchUiConfig();
        if (indexTab.folder !== folder) return;
        const relative = relativeToRoot(config.create_root, folder.destination_path);
        if (relative === null) {
            note.textContent = `This folder is outside the library root (${config.create_root}): the new path below is taken relative to that root.`;
            input.value = '';
        } else {
            note.textContent = `Relative to ${config.create_root}. Moves the directory on disk and rewrites the paths of its episodes.`;
            input.value = relative;
        }
    } catch (e) {
        note.textContent = 'Cannot read the library root: ' + e.message;
    }
}

async function moveIndexFolder() {
    const folder = indexTab && indexTab.folder;
    if (!folder) { showToast('Select a folder first', 'error'); return; }
    const newPath = document.getElementById('index-move-path').value.trim().replace(/^\/+/, '').replace(/\/+$/, '');
    if (!newPath) { showToast('Enter the new path, relative to the library root', 'error'); return; }
    let config;
    try { config = await fetchUiConfig(); } catch (e) { showToast('Cannot read the library root: ' + e.message, 'error'); return; }
    const target = normalizePath(`${config.create_root}/${newPath}`);
    if (!confirm(`Move this folder on disk?\n\n${folder.destination_path}\n→ ${target}`)) return;

    const button = document.getElementById('index-move-btn');
    button.disabled = true;
    try {
        const result = await api.moveFolder(folder.id, newPath);
        showToast(`${result.message}: ${result.episodes_updated} episode path${result.episodes_updated === 1 ? '' : 's'} updated`, 'success', 5000);
        await selectIndexFolderById(folder.id);
        if (!document.getElementById('index-register-content').classList.contains('hidden')) {
            loadIndexRegisterBrowser(indexTab.registerDir);
        }
    } catch (e) {
        showToast(e.message, 'error', 8000);
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
        loadExistingRulesInto('index-existing-content', () => loadFolderRules(folder.id), 'No rule yet for this folder: create them below, then index.'),
        api.listFiles(folder.destination_path, 'create').catch(e => { showToast('Cannot list the folder: ' + e.message, 'error'); return []; }),
        api.getFolderEpisodes(folder.id).then(r => (r && r.episodes) || []).catch(e => { showToast('Cannot read the indexed episodes: ' + e.message, 'error'); return []; })
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
        const hit = rules.map(r => ({ rule: r, episode: extractEpisode(r.regex_pattern, name) })).find(x => x.episode !== null && /^\d+$/.test(x.episode));
        if (hit) coveredFiles.push({ name, ...hit }); else uncovered.push(name);
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

    document.getElementById('index-file-count').textContent =
        `${names.length} video file${names.length > 1 ? 's' : ''}: ${indexedFiles.length} indexed, ` +
        `${coveredFiles.length} covered by a rule, ${uncovered.length} without rule`;
    if (rebuildBoard) {
        indexTab.board.clear();
        indexTab.board.addGroups(groupByPrefix(uncovered));
    }
}

async function runIndexFolder() {
    if (!indexTab || !indexTab.folder) { showToast('Select a folder first', 'error'); return; }
    const button = document.getElementById('index-run-btn');
    button.disabled = true;
    const report = document.getElementById('index-report');
    const body = document.getElementById('index-report-content');
    try {
        const result = await api.indexFolder(indexTab.folder.id);
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

// --- 8c. Specials tab ---
//
// A special is renamed by its exact name (special_renames table), then the
// folder regex catches the new name. One card = one file: new name, and the
// rule that will pick that new name up.

let specialTab = null;

function initSpecialTab() {
    if (specialTab) { specialTab.picker.load(); loadSpecialList(); return; }
    const picker = createFolderPicker('special', (f) => {
        specialTab.folder = f;
        document.getElementById('special-work-area').classList.remove('disabled-area');
        document.getElementById('special-existing').classList.remove('hidden');
        loadExistingRulesInto('special-existing-content', () => loadFolderRules(f.id), 'No existing rules for this folder.');
    }, () => {
        specialTab.folder = null;
        document.getElementById('special-work-area').classList.add('disabled-area');
        document.getElementById('special-existing').classList.add('hidden');
    });
    specialTab = { picker, folder: null, cards: [] };
    picker.load();
    loadSpecialList();
}

async function loadSpecialList() {
    const content = document.getElementById('special-list-content');
    content.innerHTML = '';
    try {
        const response = await api.getSpecials();
        const specials = response.special_renames || [];
        if (!specials.length) content.appendChild(el('div', 'text-sm text-muted', 'No special declared yet.'));
        specials.forEach(s => {
            const row = el('div', 'rule-row');
            const left = el('div');
            left.appendChild(el('div', 'font-mono text-accent', s.file_name));
            left.appendChild(el('div', 'text-muted text-sm font-mono', `→ ${s.new_file_name}`));
            const edit = el('button', 'btn btn-sm btn-secondary', 'Edit');
            edit.onclick = () => {
                if (!specialTab.folder) { showToast('Select the folder of the show first', 'error'); return; }
                addSpecialCard(s.file_name, s.new_file_name);
            };
            row.append(left, edit);
            content.appendChild(row);
        });
    } catch (e) {
        content.appendChild(el('div', 'text-danger text-sm', `Failed to load: ${e.message}`));
    }
}

function openSpecialFileModal() {
    openFileModal((names) => names.forEach(addSpecialCard));
}

function addSpecialCard(fileName, initialNewName) {
    const container = document.getElementById('special-cards');
    const empty = container.querySelector('.empty-board');
    if (empty) empty.remove();
    const cardState = { fileName, newName: initialNewName || fileName, episodeIndex: null, regexManual: null, renameManual: null, weight: 1, existingRule: null };
    const card = el('div', 'card group-card');

    const header = el('div', 'group-header');
    const title = el('div');
    title.appendChild(el('span', 'text-muted text-xs uppercase font-bold', 'Special'));
    const savedBadge = el('span', 'badge badge-success hidden', 'special saved ✓');
    title.appendChild(savedBadge);
    const remove = el('button', 'btn-icon', '🗑️');
    remove.onclick = () => card.remove();
    header.append(title, remove);
    card.appendChild(header);

    card.appendChild(el('label', 'label', 'Incoming file name (exact match)'));
    card.appendChild(el('div', 'font-mono text-accent original-name', fileName));

    card.appendChild(el('label', 'label mt-4', 'New file name — what a rule of the selected folder must recognise'));
    const newNameInput = document.createElement('input');
    newNameInput.type = 'text';
    newNameInput.className = 'input font-mono';
    newNameInput.value = cardState.newName;
    card.appendChild(newNameInput);
    const newNameMsg = el('div', 'mt-2 text-xs validation-msg');
    card.appendChild(newNameMsg);

    const specialFooter = el('div', 'group-footer');
    const saveSpecial = el('button', 'btn btn-primary btn-sm', 'Save Special');
    specialFooter.appendChild(saveSpecial);
    card.appendChild(specialFooter);

    // The rule part: the proposal until VMSAM says an existing rule already
    // catches the new name, then that rule, editable.
    const ruleBox = el('div', 'special-rule');
    card.appendChild(ruleBox);
    const ruleInfo = el('div', 'rule-info text-sm text-muted');
    let regexInput, renameInput, weightInput, saveRule;

    function renderRule() {
        const names = [cardState.newName];
        const proposal = analyzeGroup(names, cardState.episodeIndex);
        const existing = cardState.existingRule;
        const regex = cardState.regexManual !== null ? cardState.regexManual : (existing ? existing.regex_pattern : proposal.regex);
        const rename = cardState.renameManual !== null ? cardState.renameManual : (existing ? (existing.rename_pattern || '') : proposal.rename);
        ruleBox.innerHTML = '';
        ruleBox.appendChild(ruleInfo);

        const example = el('div', 'group-example');
        example.appendChild(el('span', 'text-muted text-xs', 'Episode number in the new name — click a digit run to change: '));
        const line = el('div', 'font-mono example-line');
        const numbers = new Set(numberSlots(proposal.template));
        proposal.template.slots.forEach((slot, i) => {
            if (numbers.has(i)) {
                const chip = el('span', `num-chip${i === proposal.episodeIndex ? ' selected' : ''}`, slot.v);
                chip.onclick = () => { cardState.episodeIndex = i; cardState.regexManual = null; cardState.renameManual = null; cardState.existingRule = null; renderRule(); };
                line.appendChild(chip);
            } else {
                line.appendChild(el('span', null, slot.v));
            }
        });
        example.appendChild(line);
        ruleBox.appendChild(example);

        const grid = el('div', 'grid-2');
        const regexBox = el('div');
        regexBox.appendChild(el('label', 'label', 'Regex Pattern (Python)'));
        regexInput = document.createElement('input');
        regexInput.type = 'text';
        regexInput.className = 'input font-mono';
        regexInput.value = regex;
        regexInput.oninput = () => { cardState.regexManual = regexInput.value; refreshCheck(); };
        regexBox.appendChild(regexInput);
        const regexMsg = el('div', 'mt-2 text-xs validation-msg');
        regexBox.appendChild(regexMsg);
        const renameBox = el('div');
        renameBox.appendChild(el('label', 'label', 'Rename Pattern'));
        renameInput = document.createElement('input');
        renameInput.type = 'text';
        renameInput.className = 'input font-mono';
        renameInput.value = rename;
        renameInput.oninput = () => { cardState.renameManual = renameInput.value; refreshCheck(); };
        renameBox.appendChild(renameInput);
        const renameMsg = el('div', 'mt-2 text-xs validation-msg');
        renameBox.appendChild(renameMsg);
        grid.append(regexBox, renameBox);
        ruleBox.appendChild(grid);

        const row = el('div', 'group-footer-row');
        const weightBox = el('div');
        weightBox.appendChild(el('label', 'label', 'Weight'));
        weightInput = document.createElement('input');
        weightInput.type = 'number';
        weightInput.min = 1;
        weightInput.className = 'input text-center small-number';
        weightInput.value = existing ? existing.weight : cardState.weight;
        weightInput.oninput = () => { cardState.weight = parseInt(weightInput.value, 10) || 1; };
        weightBox.appendChild(weightInput);
        const summary = el('div', 'extraction-box');
        row.append(weightBox, summary);
        ruleBox.appendChild(row);

        const ruleFooter = el('div', 'group-footer');
        saveRule = el('button', 'btn btn-primary btn-sm', existing ? 'Update Rule' : 'Save Rule');
        saveRule.onclick = submitRule;
        ruleFooter.appendChild(saveRule);
        ruleBox.appendChild(ruleFooter);

        function refreshCheck() {
            const result = testPattern(regexInput.value, [cardState.newName]);
            const first = result.perFile[0];
            if (result.syntaxError) {
                regexMsg.textContent = '⚠ Invalid regex syntax';
                regexMsg.className = 'mt-2 text-xs text-danger validation-msg';
            } else if (first.valid) {
                regexMsg.textContent = '✓ The new name yields an episode number';
                regexMsg.className = 'mt-2 text-xs text-success validation-msg';
            } else {
                regexMsg.textContent = "⚠ No valid 'episode' group on the new name";
                regexMsg.className = 'mt-2 text-xs text-warning validation-msg';
            }
            if (!renameInput.value.includes(EPISODE_PLACEHOLDER)) {
                renameMsg.textContent = `⚠ Must contain ${EPISODE_PLACEHOLDER}`;
                renameMsg.className = 'mt-2 text-xs text-warning validation-msg';
            } else {
                renameMsg.textContent = '';
                renameMsg.className = 'mt-2 text-xs text-muted validation-msg';
            }
            summary.innerHTML = '';
            summary.appendChild(el('span', 'text-xs text-muted uppercase', 'Extracted episode'));
            summary.appendChild(el('span', `text-3xl font-bold ${first.valid ? 'text-success' : 'text-muted'}`, first.valid ? first.episode : '-'));
        }
        refreshCheck();
    }

    function setRuleInfo(text, tone) {
        ruleInfo.textContent = text;
        ruleInfo.className = `rule-info text-sm ${tone || 'text-muted'}`;
    }

    newNameInput.oninput = () => {
        cardState.newName = newNameInput.value;
        cardState.episodeIndex = null;
        cardState.regexManual = null;
        cardState.renameManual = null;
        cardState.existingRule = null;
        savedBadge.classList.add('hidden');
        card.classList.remove('saved');
        if (cardState.newName === cardState.fileName) {
            newNameMsg.textContent = '⚠ The new name must differ from the incoming name';
            newNameMsg.className = 'mt-2 text-xs text-warning validation-msg';
        } else if (/[\/\\]/.test(cardState.newName)) {
            newNameMsg.textContent = '⚠ A file name, not a path';
            newNameMsg.className = 'mt-2 text-xs text-danger validation-msg';
        } else {
            newNameMsg.textContent = '';
        }
        setRuleInfo('Save the special first: VMSAM answers whether a rule of this folder already catches the new name.');
        renderRule();
    };

    saveSpecial.onclick = async () => {
        if (!specialTab.folder) { showToast('Select a folder first', 'error'); return; }
        const newName = cardState.newName.trim();
        if (!newName || newName === cardState.fileName) { showToast('Give the special a new name first', 'error'); return; }
        saveSpecial.disabled = true;
        try {
            const special = await api.createSpecial({ file_name: cardState.fileName, new_file_name: newName, destination_path: specialTab.folder.destination_path });
            showToast(special.message, 'success');
            savedBadge.classList.remove('hidden');
            card.classList.add('saved');
            cardState.regexManual = null;
            cardState.renameManual = null;
            if (special.matching_regex) {
                cardState.existingRule = special.matching_regex;
                setRuleInfo(`An existing rule of this folder already catches the new name (episode ${special.matching_regex.extracted_episode}). It is shown below; edit it only if needed.`, 'text-success');
            } else {
                cardState.existingRule = null;
                setRuleInfo('No rule catches the new name yet: save the rule below so the renamed file is integrated.', 'text-warning');
            }
            renderRule();
            loadSpecialList();
        } catch (e) {
            // A 400 is VMSAM's control: wrong folder, path in a name, same name. Nothing was stored.
            showToast('Special not saved: ' + e.message, 'error', 8000);
        } finally {
            saveSpecial.disabled = false;
        }
    };

    async function submitRule() {
        if (!specialTab.folder) { showToast('Select a folder first', 'error'); return; }
        const newName = cardState.newName.trim();
        const regex = regexInput.value;
        const rename = renameInput.value;
        const check = testPattern(regex, [newName]);
        if (!check.allValid) { showToast('The regex must extract a valid episode number from the new name', 'error'); return; }
        if (!rename.includes(EPISODE_PLACEHOLDER)) { showToast(`Rename pattern must contain ${EPISODE_PLACEHOLDER}`, 'error'); return; }
        saveRule.disabled = true;
        try {
            const rule = await api.createRegex({ regex_pattern: regex, rename_pattern: rename, weight: parseInt(weightInput.value, 10) || 1, example_filename: newName, destination_path: specialTab.folder.destination_path });
            showToast(rule.message, 'success');
            cardState.existingRule = { regex_pattern: regex, rename_pattern: rename, weight: parseInt(weightInput.value, 10) || 1 };
            setRuleInfo('Rule saved: the renamed file will be integrated into this folder.', 'text-success');
            renderRule();
            loadExistingRulesInto('special-existing-content', () => loadFolderRules(specialTab.folder.id), 'No existing rules for this folder.');
        } catch (e) {
            showToast('Rule not saved: ' + e.message, 'error', 8000);
        } finally {
            saveRule.disabled = false;
        }
    }

    newNameInput.oninput();
    container.appendChild(card);
}

// --- 8d. Incrementaller tab ---

let incrementallerTab = null;

function initIncrementallerTab() {
    if (incrementallerTab) { loadIncrementallerList(); return; }
    const board = createGroupBoard({
        containerId: 'incr-cards', mode: 'incrementaller',
        emptyText: 'Add files: one rule per file by default, drag a file onto another rule to group them.'
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

// Exposed for the engine self-test page (no module system, plain script).
window.vmsamEngine = { tokenizeName, buildTemplate, analyzeGroup, testPattern, groupByPrefix, extractEpisode, buildRegex };
