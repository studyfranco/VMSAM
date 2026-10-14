// VMSAM web interface — Movies section. Vanilla JS, loaded after app.js and
// reusing its helpers (api client helpers, el, showToast, confirmDialog,
// filename engine, folder browser patterns, file modal).
//
// Tabs live in <section id="movie-...-tab"> elements of index.html (between
// the "Movie tabs" comment and MOVIES-SECTIONS-END) and their nav buttons in
// the nav with data-section="movies". Each tab registers its init in
// TAB_INIT (defined in app.js).
//
// A movie is one kept file (a film or a show special). Its `title` is its
// NAME in VMSAM (globally unique, the key of `movies`), not the file name.
// The file name comes from the rule that matched it: the rule's rename
// pattern holds `{<suffix>}`, replaced by the movie's suffix
// (`Movie A (2001){<suffix>}.mkv`); a rule without rename pattern keeps the
// downloaded name. The file lands in the movie's `destination_path`. Rules
// have no episode group, they only have to match the downloaded names.
// Every call goes through the relay /api/movie/<name>; a refusal is shown
// with VMSAM's own `detail`.
//
// Sections:
//   M1. API client and caches
//   M2. helpers (names, regex proposal, paths)
//   M3. shared widgets: search picker, directory browser, metadata generator
//   M4. tabs: creation, bulk regex, index, move, movies, rules, errors

// --- M1. API client and caches ---

const movieApi = {
    get(name, params) {
        const query = params ? '?' + new URLSearchParams(params).toString() : '';
        return apiFetch(`/api/movie/${name}${query}`);
    },
    post(name, payload) { return postJson(`/api/movie/${name}`, payload); }
};

// VMSAM answers 404 for an empty list: an empty array here. Any other
// refusal keeps VMSAM's own `detail`.
async function movieListOr404(name, key, params) {
    const query = params ? '?' + new URLSearchParams(params).toString() : '';
    const res = await fetch(`/api/movie/${name}${query}`);
    if (res.status === 404) return [];
    const text = await res.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (e) { payload = null; }
    if (!res.ok) {
        const detail = payload && payload.detail ? payload.detail : (text || `HTTP ${res.status}`);
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return (payload && payload[key]) || [];
}

let movieListCache = null;

// Movies are keyed `movie_id` (rules and errors too).
async function fetchMovies(force = false) {
    if (movieListCache && !force) return movieListCache;
    movieListCache = await movieListOr404('movies_list', 'movies');
    return movieListCache;
}

function fetchMovieFolders() {
    return movieListOr404('folders_list', 'folders');
}

// regex_movie answers 404 for an unknown movie too: the callers know it exists.
function fetchMovieRules(movieId) {
    return movieListOr404('regex_movie', 'regex_patterns', { movie_id: movieId });
}

async function fetchAllMovieRules() {
    const response = await movieApi.get('regex_list');
    return (response && response.regex_patterns) || [];
}

let movieMetadataCache = null;

async function fetchMovieMetadataConfig() {
    if (!movieMetadataCache) movieMetadataCache = await movieApi.get('metadata_providers');
    return movieMetadataCache;
}

// After any change of a movie: every list is fetched again when next needed.
function invalidateMovies() {
    movieListCache = null;
}

// --- M2. Helpers ---

const MOVIE_SUFFIX_PLACEHOLDER = '{<suffix>}';

// The name a rule gives a file: its rename pattern with `{<suffix>}` replaced
// by the movie's suffix; no rename pattern = the file keeps its name.
function movieRenamedName(renamePattern, suffix, fileName) {
    if (!renamePattern) return fileName || '';
    return renamePattern.split(MOVIE_SUFFIX_PLACEHOLDER).join(suffix || '');
}

// null when the rename pattern can be sent, else the reason: empty is
// allowed (the file keeps its name), otherwise it must hold `{<suffix>}` and
// no `/` (a file name, the directory is the movie's).
function movieRenameProblem(renamePattern) {
    const text = String(renamePattern || '');
    if (!text) return null;
    if (!text.includes(MOVIE_SUFFIX_PLACEHOLDER)) return `must contain ${MOVIE_SUFFIX_PLACEHOLDER} (or be empty: the file keeps its name)`;
    if (text.includes('/')) return 'a file name cannot hold a /';
    if (text.trim() !== text) return 'no leading or trailing space';
    return null;
}

// The rename pattern proposed for a group of names: the first name (its
// spelling) with the wild spans of the engine's template dropped (the parts
// that differ between the names: release tags, groups), the separators on
// both sides of a hole merged, then `{<suffix>}` right before the extension:
// `Movie.G.2005.1080p.BluRay-AAA.mkv` + `Movie.G.2005.720p.WEB-BBB.mkv` ->
// `Movie.G.2005.1080p{<suffix>}.mkv`.
function movieRenameFor(names) {
    if (!names.length) return '';
    let base = names[0];
    if (names.length > 1) {
        const GAP = '\u0000';
        let out = '';
        buildTemplate(names).slots.forEach(slot => {
            if (slot.wild) { if (!out.endsWith(GAP)) out += GAP; return; }
            out += slot.v;
        });
        // a bracket pair that only held a dropped span goes with it
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
        if (out) base = out;
    }
    const extension = movieFileExtension(base);
    return base.slice(0, base.length - extension.length) + MOVIE_SUFFIX_PLACEHOLDER + extension;
}

function movieFileExtension(name) {
    const match = String(name || '').match(/\.[A-Za-z0-9]{1,5}$/);
    return match ? match[0] : '';
}

function movieBaseName(path) {
    return String(path || '').split('/').filter(Boolean).pop() || '';
}

// Absolute directory -> path shown to the user, relative to CREATE_ROOT
// (fetchUiConfig must have run). Outside the root: the absolute path.
function movieRel(absolute) {
    const root = uiConfigCache ? uiConfigCache.create_root : null;
    const relative = root ? relativeToRoot(root, absolute) : null;
    if (relative === null) return absolute;
    return relative || '/ (library root)';
}

// Where a directory picker opens for a selected folder: its parent, relative
// to CREATE_ROOT ('' = the root, also when nothing is selected).
function movieParentDir(folder) {
    const root = uiConfigCache ? uiConfigCache.create_root : null;
    const relative = folder && root ? relativeToRoot(root, folder.destination_path) : null;
    return relative ? relative.split('/').slice(0, -1).join('/') : '';
}

// A movie is named by its title (its unique name in VMSAM).
function movieLabel(movie) {
    return String(movie.title || '');
}

// A title guessed from a file name: extension and bracket tags dropped, dots
// and underscores become spaces when the name has no space, and the name is
// cut at the year (kept in parentheses) or at the first technical tag.
function guessMovieTitle(name) {
    let text = String(name || '').replace(/\.[A-Za-z0-9]{1,5}$/, '');
    text = text.replace(/\[[^\]]*\]|\{[^}]*\}/g, ' ');
    if (!/\s/.test(text.trim())) text = text.replace(/[._]+/g, ' ');
    const year = text.match(/^(.*?\S)[\s(]+((?:19|20)\d\d)\b/);
    if (year) return `${year[1].trim()} (${year[2]})`;
    const cut = text.search(/\b(?:\d{3,4}p|x26[45]|h\.?26[45]|web|bluray|bdrip|remux|multi|vostfr)\b/i);
    if (cut > 0) text = text.slice(0, cut);
    return text.replace(/[\s\-]+$/, '').replace(/\s+/g, ' ').trim();
}

// True when the (Python) pattern matches the name in the browser; a pattern
// that does not compile matches nothing.
function movieRegexMatches(pattern, name) {
    try { return toJsRegex(pattern).test(name); } catch (e) { return false; }
}

// A file name a movie regex matches, for the example VMSAM requires when a
// rule changes weight or movie: sampleFromRegex (app.js) on the pattern, or
// on each alternative of a `^(?:a|b)$` rule; null when none fits (the user
// then types a real file name).
function movieSampleFromRegex(pattern) {
    const text = String(pattern || '');
    const candidates = [sampleFromRegex(text)];
    const alternatives = text.match(/^\^\(\?:(.*)\)\$$/);
    if (alternatives) {
        // split on the | neither escaped nor nested
        const body = alternatives[1];
        let depth = 0;
        let from = 0;
        for (let i = 0; i <= body.length; i++) {
            const c = body[i];
            if (c === '\\') { i++; continue; }
            if (c === '(') depth++;
            else if (c === ')') depth--;
            else if ((c === '|' && depth === 0) || i === body.length) {
                candidates.push(sampleFromRegex('^' + body.slice(from, i) + '$'));
                from = i + 1;
            }
        }
    }
    return candidates.find(name => name && movieRegexMatches(text, name)) || null;
}

// { syntaxError, perFile: [{ name, match }], allMatch, hasGroup }
function checkMovieRegex(pattern, names) {
    let syntaxError = null;
    try { toJsRegex(pattern); } catch (e) { syntaxError = e.message; }
    if (!pattern) syntaxError = 'empty pattern';
    const perFile = names.map(name => ({ name, match: !syntaxError && movieRegexMatches(pattern, name) }));
    return {
        syntaxError,
        perFile,
        allMatch: !syntaxError && perFile.every(f => f.match),
        hasGroup: /\(\?P<episode>/.test(pattern || '')
    };
}

// The rule proposed for a group of names, with no episode group: one file
// gives its exact name anchored; several the engine's template (literals
// escaped, a number that only changes value \d+, anything else .+). Should
// the template miss a name, the names are listed as alternatives.
function movieRegexFor(names) {
    if (!names.length) return '';
    if (names.length === 1) return '^' + escapeRegex(names[0]) + '$';
    const pattern = buildRegex(buildTemplate(names), null);
    if (checkMovieRegex(pattern, names).allMatch) return pattern;
    return '^(?:' + names.map(escapeRegex).join('|') + ')$';
}

function movieMessage(node, text, kind) {
    node.textContent = text || '';
    node.className = `movie-message ${kind ? 'text-' + kind : 'text-muted'}`;
    node.classList.toggle('hidden', !text);
}

function movieInput(type, className, value) {
    const input = document.createElement('input');
    input.type = type;
    input.className = className || 'input';
    if (value !== undefined && value !== null) input.value = value;
    input.spellcheck = false;
    input.autocomplete = 'off';
    return input;
}

function movieButton(text, className, onClick, title) {
    const button = el('button', className || 'btn btn-secondary btn-sm', text);
    button.type = 'button';
    if (title) button.title = title;
    if (onClick) button.onclick = onClick;
    return button;
}

function movieLabelled(text, input) {
    const box = el('div', 'form-group');
    box.append(el('label', 'label', text), input);
    return box;
}

function movieTabShown(tabId) {
    const node = document.getElementById(tabId);
    return !!node && !node.classList.contains('hidden');
}

// --- M3. Shared widgets ---

// A search box over a list loaded on demand, with a dropdown and a "selected"
// line. options: placeholder, emptyText, load() -> items, text(item)
// (searched), render(item) -> node (dropdown row), chosen(item) -> text,
// onSelect(item), onClear(); optional rank(item, query) -> number, the
// lowest listed first (stable). picker.select(item, true) selects silently.
function createMovieSearchPicker(host, options) {
    host.innerHTML = '';
    const input = movieInput('search', 'input movie-picker-input');
    input.placeholder = options.placeholder || 'Search…';
    const list = el('div', 'dir-browser-container movie-picker-list hidden');
    const chosen = el('div', 'hidden flex flex-between items-center p-2 selected-folder movie-picker-chosen');
    const chosenText = el('span', 'font-mono text-accent');
    const picker = { selected: null, items: [], input };
    chosen.append(chosenText, movieButton('Change', 'btn btn-sm btn-secondary', () => picker.clear()));
    host.append(input, list, chosen);

    const render = () => {
        list.innerHTML = '';
        const query = input.value.trim().toLowerCase();
        let items = picker.items.filter(item => !query || options.text(item).toLowerCase().includes(query));
        if (query && options.rank) items = items.map((item, i) => [options.rank(item, query), i, item]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(x => x[2]);
        if (!items.length) list.appendChild(el('div', 'text-muted text-sm p-2', picker.items.length ? 'Nothing matches' : (options.emptyText || 'Nothing yet')));
        items.slice(0, 300).forEach(item => {
            const row = el('div', 'dir-item movie-picker-item');
            row.appendChild(options.render(item));
            row.onclick = () => picker.select(item);
            list.appendChild(row);
        });
    };

    picker.load = async () => {
        try {
            picker.items = await options.load();
        } catch (e) {
            picker.items = [];
            list.innerHTML = '';
            list.appendChild(el('div', 'text-danger text-sm p-2', e.message));
            return;
        }
        render();
    };
    picker.select = (item, silent) => {
        picker.selected = item;
        chosenText.textContent = options.chosen(item);
        chosen.classList.remove('hidden');
        input.classList.add('hidden');
        list.classList.add('hidden');
        input.value = '';
        if (!silent && options.onSelect) options.onSelect(item);
    };
    picker.clear = () => {
        picker.selected = null;
        chosen.classList.add('hidden');
        input.classList.remove('hidden');
        if (options.onClear) options.onClear();
    };
    input.addEventListener('focus', async () => {
        list.classList.remove('hidden');
        await picker.load();
    });
    input.addEventListener('input', () => { list.classList.remove('hidden'); render(); });
    return picker;
}

// One listener closes every open picker dropdown on a click outside its picker.
document.addEventListener('click', (e) => {
    document.querySelectorAll('.movie-picker-list:not(.hidden)').forEach(list => {
        if (!list.parentElement.contains(e.target)) list.classList.add('hidden');
    });
});

function moviePickerRow(movie) {
    const row = el('div', 'movie-picker-row');
    row.appendChild(el('span', 'movie-picker-title', movieLabel(movie)));
    row.appendChild(el('span', 'text-muted text-xs font-mono', `${movie.suffix ? 'suffix "' + movie.suffix + '" · ' : ''}${movieRel(movie.destination_path)} · #${movie.movie_id}`));
    return row;
}

// Movies are searched by title first: the movies whose title holds the text
// come first (a title starting with it before), then those matched by their
// directory, suffix or id.
function movieTitleRank(movie, query) {
    const title = movieLabel(movie).toLowerCase();
    if (title === query) return 0;
    if (title.startsWith(query)) return 1;
    if (title.includes(query)) return 2;
    return 3;
}

function createMoviePicker(host, onSelect, onClear) {
    return createMovieSearchPicker(host, {
        placeholder: 'Search a movie by title (or directory, suffix, id)…',
        emptyText: 'No movie yet: create one first',
        load: () => fetchMovies(),
        text: m => `${movieLabel(m)} ${m.suffix || ''} ${m.destination_path} #${m.movie_id}`,
        rank: movieTitleRank,
        render: moviePickerRow,
        chosen: m => `${movieLabel(m)} — ${movieRel(m.destination_path)} (#${m.movie_id})`,
        onSelect,
        onClear
    });
}

function createMovieFolderPicker(host, onSelect, onClear) {
    return createMovieSearchPicker(host, {
        placeholder: 'Search a directory holding movies…',
        emptyText: 'No movie directory yet',
        load: () => fetchMovieFolders(),
        text: f => f.destination_path,
        render: f => {
            const row = el('div', 'movie-picker-row');
            row.appendChild(el('span', 'font-mono', movieRel(f.destination_path)));
            row.appendChild(el('span', 'badge', `${f.movies_count} movie${f.movies_count === 1 ? '' : 's'}`));
            return row;
        },
        chosen: f => movieRel(f.destination_path),
        onSelect,
        onClear
    });
}

// Directory pickers: app.js's large modal (openDirectoryPicker) over
// CREATE_ROOT, with its name filter. movieDirPicker(options): title, hint,
// start, chooseLabel, onPick(path); `blocked(path)` -> null or the reason a
// directory cannot be chosen: such a row opens instead of being selected, and
// the listed directory cannot be picked with the "here" button.
function movieDirPicker(options) {
    const blocked = options.blocked || (() => null);
    return openDirectoryPicker({
        title: options.title,
        hint: options.hint,
        startPath: cleanRelativePath(options.start || ''),
        chooseLabel: options.chooseLabel || 'Choose',
        hereLabel: options.hereLabel || options.chooseLabel || 'Choose this directory',
        allowCurrent: path => !blocked(path),
        decorate: (row, item) => {
            if (options.decorate) options.decorate(row, item);
            const reason = blocked(item.path);
            if (!reason) return;
            row.classList.add('movie-dir-blocked');
            row.title = reason + ': a click opens it';
            row.onclick = () => { if (dirPicker) dirPicker.load(item.path); };
            row.onkeydown = (e) => { if ((e.key === 'Enter' || e.key === 'ArrowRight') && dirPicker) { e.preventDefault(); dirPicker.load(item.path); } };
        },
        onPick: path => {
            const reason = blocked(path);
            if (reason) { showToast(reason, 'error'); return; }
            options.onPick(path);
        }
    });
}

// The picker over the directories holding movies (folders_list): those rows
// carry their movie count, a directory with movies further down says so, and
// only a directory holding movies can be chosen (a click on another opens
// it). onChoose(folder).
async function openMovieFolderModal(title, onChoose, start) {
    let folders, config;
    try {
        [folders, config] = await Promise.all([fetchMovieFolders(), fetchUiConfig()]);
    } catch (e) {
        showToast('Directories not loaded: ' + e.message, 'error', 8000);
        return null;
    }
    const byPath = new Map();
    const above = new Set();
    folders.forEach(folder => {
        const relative = relativeToRoot(config.create_root, folder.destination_path);
        if (relative === null) return;
        byPath.set(relative, folder);
        const parts = relative.split('/');
        for (let i = 1; i < parts.length; i++) above.add(parts.slice(0, i).join('/'));
    });
    const outside = folders.length - byPath.size;
    const count = n => `${n} movie${n === 1 ? '' : 's'}`;
    return movieDirPicker({
        title,
        hint: `Directories holding movies show their count; "movies inside" marks a directory with movies further down.${outside ? ` ${outside} director${outside > 1 ? 'ies' : 'y'} outside the library root: use the search box instead.` : ''}`,
        start,
        chooseLabel: 'Select',
        hereLabel: 'Select this directory',
        blocked: path => byPath.has(path) ? null : 'No movie directly in this directory',
        decorate: (row, item) => {
            const name = row.querySelector('.dir-name');
            const folder = byPath.get(item.path);
            if (folder) name.appendChild(el('span', 'badge badge-success movie-dir-count', count(folder.movies_count)));
            else if (above.has(item.path)) name.appendChild(el('span', 'badge movie-dir-below', 'movies inside'));
        },
        onPick: path => onChoose(byPath.get(path))
    });
}

// Metadata id generator. Apps (plex, jellyfin…) and providers come from
// GET metadata_providers; one input per provider, enabled when a ticked app
// knows it; the tag string comes from POST metadata_tags (VMSAM builds it,
// the UI never guesses the brackets). actions: [{ text, title, use(tags) }].
// onChange() is called when an id changes.
function createMovieMetadataGenerator(host, actions, onChange) {
    const gen = { tags: '', inputs: new Map(), apps: new Map(), sequence: 0, ready: false };
    host.innerHTML = '';
    const loading = el('div', 'text-muted text-sm', 'Loading the metadata providers…');
    host.appendChild(loading);
    const appsRow = el('div', 'movie-meta-apps');
    const providersBox = el('div', 'movie-meta-providers');
    const output = el('div', 'movie-meta-output');
    const tagsLine = el('code', 'font-mono movie-meta-tags');
    const perApp = el('div', 'movie-meta-per-app text-xs');
    const unsupported = el('div', 'movie-meta-unsupported text-xs');
    const errorLine = el('div', 'movie-message hidden');
    const buttons = el('div', 'movie-meta-actions');
    const actionButtons = actions.map(action => {
        const button = movieButton(action.text, 'btn btn-secondary btn-sm', () => { if (gen.tags) action.use(gen.tags); }, action.title);
        buttons.appendChild(button);
        return button;
    });
    output.append(el('div', 'label', 'Generated tags'), tagsLine, perApp, unsupported, errorLine, buttons);
    let config = null;
    let timer = null;

    const selectedApps = () => Array.from(gen.apps.entries()).filter(([, box]) => box.checked).map(([app]) => app);

    const addProviderRow = (provider, known) => {
        const row = el('div', 'movie-meta-provider');
        const input = movieInput('text', 'input font-mono movie-meta-input');
        input.placeholder = 'id';
        input.dataset.provider = provider;
        input.oninput = () => { schedule(); if (onChange) onChange(); };
        const support = el('div', 'movie-meta-support text-xs');
        row.append(el('label', 'movie-meta-name font-mono', provider), input, support);
        if (!known) support.appendChild(el('span', 'text-warning', 'not in the providers file: stored, no tag'));
        providersBox.appendChild(row);
        gen.inputs.set(provider, { input, support, known, row });
    };

    const refreshSupport = () => {
        const apps = selectedApps();
        gen.inputs.forEach((entry, provider) => {
            if (!entry.known) return;
            const tags = config.providers[provider] || {};
            const supported = apps.filter(app => tags[app] !== undefined);
            entry.input.disabled = supported.length === 0;
            entry.row.classList.toggle('movie-meta-off', supported.length === 0);
            entry.support.innerHTML = '';
            apps.forEach(app => {
                const ok = tags[app] !== undefined;
                entry.support.appendChild(el('span', `movie-meta-chip ${ok ? 'ok' : 'no'}`, ok ? `${app}: ${tags[app]}` : `${app}: unsupported`));
            });
            if (!apps.length) entry.support.appendChild(el('span', 'text-muted', 'tick an app'));
        });
    };

    // Ids of the enabled inputs: those the tags are built from.
    const tagIds = () => {
        const ids = {};
        gen.inputs.forEach((entry, provider) => {
            const value = entry.input.value.trim();
            if (value && !entry.input.disabled) ids[provider] = value;
        });
        return ids;
    };

    const setTags = (tags) => {
        gen.tags = tags;
        tagsLine.textContent = tags || '(no tag: type an id and tick an app)';
        tagsLine.classList.toggle('text-muted', !tags);
        actionButtons.forEach(b => { b.disabled = !tags; });
    };

    const compute = async () => {
        const sequence = ++gen.sequence;
        const apps = selectedApps();
        const ids = tagIds();
        perApp.innerHTML = '';
        unsupported.innerHTML = '';
        movieMessage(errorLine, '');
        if (!apps.length || !Object.keys(ids).length) { setTags(''); gen.ready = true; return; }
        try {
            const result = await movieApi.post('metadata_tags', { apps, ids });
            if (sequence !== gen.sequence) return;
            setTags(result.tags || '');
            Object.entries(result.per_app || {}).forEach(([app, text]) => {
                const line = el('div', 'movie-meta-app-line');
                line.append(el('span', 'text-muted', `${app}: `), el('span', 'font-mono', text || '(none)'));
                perApp.appendChild(line);
            });
            (result.unsupported || []).forEach(([app, provider]) => {
                unsupported.appendChild(el('div', 'text-warning', `${app} has no tag for ${provider}`));
            });
        } catch (e) {
            if (sequence !== gen.sequence) return;
            setTags('');
            movieMessage(errorLine, e.message, 'danger');
        }
        gen.ready = true;
    };

    const schedule = () => {
        gen.ready = false;
        clearTimeout(timer);
        timer = setTimeout(compute, 200);
    };

    // Every non-empty id typed, enabled or not: what the movie stores.
    gen.getIds = () => {
        const ids = {};
        gen.inputs.forEach((entry, provider) => {
            const value = entry.input.value.trim();
            if (value) ids[provider] = value;
        });
        return ids;
    };
    gen.refresh = schedule;

    gen.whenLoaded = fetchMovieMetadataConfig().then(loaded => {
        config = loaded;
        loading.remove();
        appsRow.appendChild(el('span', 'label movie-meta-apps-label', 'Apps'));
        Object.keys(config.brackets || {}).forEach(app => {
            const label = el('label', 'movie-meta-app');
            const box = document.createElement('input');
            box.type = 'checkbox';
            box.className = 'input-checkbox';
            box.checked = true;
            box.value = app;
            box.onchange = () => { refreshSupport(); schedule(); };
            const brackets = config.brackets[app];
            label.append(box, el('span', null, app), el('span', 'text-muted text-xs font-mono', `${brackets.begin}…${brackets.end}`));
            appsRow.appendChild(label);
            gen.apps.set(app, box);
        });
        Object.keys(config.providers || {}).forEach(provider => addProviderRow(provider, true));
        host.append(appsRow, providersBox, output);
        refreshSupport();
        setTags('');
        gen.ready = true;
    }).catch(e => {
        loading.className = 'text-danger text-sm';
        loading.textContent = 'Metadata providers unavailable: ' + e.message;
    });

    gen.setIds = (ids) => gen.whenLoaded.then(() => {
        gen.inputs.forEach(entry => { entry.input.value = ''; });
        Object.entries(ids || {}).forEach(([provider, value]) => {
            if (!gen.inputs.has(provider)) addProviderRow(provider, false);
            gen.inputs.get(provider).input.value = value;
        });
        refreshSupport();
        schedule();
    });
    return gen;
}

// --- M4a. Movie Creation tab ---

let movieCreateTab = null;

function movieCreateValues() {
    return {
        dest: cleanRelativePath(document.getElementById('movie-create-dest').value),
        title: document.getElementById('movie-create-title').value,
        suffix: document.getElementById('movie-create-suffix').value
    };
}

function openMovieCreateBrowser() {
    movieDirPicker({
        title: 'Destination directory of the movie',
        hint: 'Select the directory the movie goes in (several movies may share one). A sub-directory that does not exist yet can be typed afterwards: VMSAM creates it.',
        start: document.getElementById('movie-create-dest').value,
        chooseLabel: 'Use this directory',
        onPick: path => setMovieCreateField('movie-create-dest', path)
    });
}

function initMovieCreateTab() {
    if (movieCreateTab) { checkMovieCreateTitle(); return; }
    const meta = createMovieMetadataGenerator(document.getElementById('movie-create-meta'), [
        { text: 'Use as suffix', title: 'Suffix = a space and the tags', use: tags => setMovieCreateField('movie-create-suffix', ' ' + tags) },
        { text: 'Append to suffix', title: 'Adds the tags at the end of the suffix', use: tags => setMovieCreateField('movie-create-suffix', appendMovieTags(document.getElementById('movie-create-suffix').value, tags)) },
        {
            text: 'Append to the directory name',
            title: 'Adds the tags at the end of the last directory of the destination',
            use: tags => {
                const dest = cleanRelativePath(document.getElementById('movie-create-dest').value);
                if (!dest) { showToast('Choose or type the destination directory first', 'error'); return; }
                setMovieCreateField('movie-create-dest', appendMovieTags(dest, tags).trim());
            }
        }
    ], updateMovieCreatePreview);
    movieCreateTab = { meta, returnGroupId: null, existing: null, checkedTitle: null, titleTimer: null, titleSequence: 0 };
    ['movie-create-dest', 'movie-create-suffix'].forEach(id => document.getElementById(id).addEventListener('input', updateMovieCreatePreview));
    document.getElementById('movie-create-title').addEventListener('input', () => {
        clearTimeout(movieCreateTab.titleTimer);
        movieCreateTab.titleTimer = setTimeout(checkMovieCreateTitle, 250);
        updateMovieCreatePreview();
    });
    document.getElementById('movie-create-browse').onclick = openMovieCreateBrowser;
    document.getElementById('movie-create-btn').onclick = submitMovieCreate;
    document.getElementById('movie-create-back').onclick = () => switchTab('movie-regex-tab');
    fetchUiConfig().then(config => { document.getElementById('movie-create-root').textContent = config.create_root; })
        .catch(e => showToast('Cannot read the library root: ' + e.message, 'error'));
    checkMovieCreateTitle();
}

// "a {x}" + "{x} [y]" -> "a {x} [y]"; tags already at the end are not repeated.
function appendMovieTags(text, tags) {
    const base = String(text || '').replace(/\s+$/, '');
    if (base.endsWith(tags)) return base;
    return base ? `${base} ${tags}` : ` ${tags}`;
}

function setMovieCreateField(id, value) {
    const input = document.getElementById(id);
    input.value = value;
    input.dispatchEvent(new Event('input'));
}

// The movie of a title (GET movie?title=), null when none: `movies` creates
// or updates by title, so the tab says which it will do.
async function fetchMovieByTitle(title) {
    const res = await fetch('/api/movie/movie?' + new URLSearchParams({ title }).toString());
    if (res.status === 404) return null;
    const text = await res.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (e) { payload = null; }
    if (!res.ok) throw new Error(payload && payload.detail ? String(payload.detail) : (text || `HTTP ${res.status}`));
    return payload;
}

async function checkMovieCreateTitle() {
    const title = document.getElementById('movie-create-title').value;
    const sequence = ++movieCreateTab.titleSequence;
    if (!title.trim()) {
        movieCreateTab.existing = null;
        movieCreateTab.checkedTitle = title;
        updateMovieCreatePreview();
        return;
    }
    let existing = null;
    try {
        existing = await fetchMovieByTitle(title);
    } catch (e) {
        existing = null;
    }
    if (sequence !== movieCreateTab.titleSequence) return;
    movieCreateTab.existing = existing && existing.title === title ? existing : null;
    movieCreateTab.checkedTitle = title;
    updateMovieCreatePreview();
}

function updateMovieCreatePreview() {
    const preview = document.getElementById('movie-create-preview');
    const status = document.getElementById('movie-create-title-status');
    const button = document.getElementById('movie-create-btn');
    const { dest, title, suffix } = movieCreateValues();
    const tab = movieCreateTab;
    const existing = tab && tab.checkedTitle === title ? tab.existing : null;
    const problems = [];
    if (!title.trim()) problems.push('the title is empty');
    if (/\//.test(suffix)) problems.push('the suffix cannot hold a / (it goes in a file name)');
    if (dest.split('/').includes('..')) problems.push('".." is refused in the directory');
    status.innerHTML = '';
    if (existing) {
        const existingDir = movieRel(existing.destination_path);
        status.className = 'text-xs mt-1 text-warning movie-create-exists';
        status.textContent = `A movie has this title (#${existing.movie_id}, ${existingDir}): saving updates it.`;
        const root = uiConfigCache ? uiConfigCache.create_root : null;
        const relative = root ? relativeToRoot(root, existing.destination_path) : null;
        if (existing.file_path && relative !== dest) {
            problems.push(`#${existing.movie_id} has a file in ${existingDir}: its directory is changed in the Movies tab (Edit, the file is moved)`);
        }
    } else if (title.trim() && tab && tab.checkedTitle === title) {
        status.className = 'text-xs mt-1 text-success movie-create-new';
        status.textContent = 'New title: a movie is created.';
    } else {
        status.className = 'text-xs mt-1';
    }
    preview.innerHTML = '';
    const line = (key, ...nodes) => {
        const row = el('div');
        row.append(el('span', 'text-muted', key + ': '), ...nodes);
        preview.appendChild(row);
    };
    line('Movie', el('span', 'text-main', title || '<title>'), el('span', 'text-muted', existing ? ` (update of #${existing.movie_id})` : ' (new)'));
    line('Directory', el('span', 'font-mono text-main', (dest || '(library root)') + '/'));
    line('File name', el('span', 'font-mono text-muted', `…${MOVIE_SUFFIX_PLACEHOLDER}.<ext>`), el('span', 'text-muted', ' → '),
        el('span', 'font-mono text-success movie-create-suffix-shown', `…${suffix}.<ext>`), el('span', 'text-muted', ' (from the rename pattern of the rule that matches the file)'));
    if (problems.length) preview.appendChild(el('div', 'text-warning text-xs mt-1', '⚠ ' + problems.join('; ')));
    button.textContent = existing ? `Update movie #${existing.movie_id}` : 'Create Movie';
    button.disabled = problems.length > 0;
}

// Opens the creation tab with a title (from a Bulk Regex card); the movie
// created next is linked to that card.
function startMovieCreation(title, groupId) {
    switchTab('movie-create-tab');
    setMovieCreateField('movie-create-title', title || '');
    checkMovieCreateTitle();
    movieCreateTab.returnGroupId = groupId === undefined ? null : groupId;
    document.getElementById('movie-create-back').classList.toggle('hidden', movieCreateTab.returnGroupId === null);
    document.getElementById('movie-create-title').focus();
}

// `movies` creates or updates by title (database only), full data.
async function submitMovieCreate() {
    const { dest, title, suffix } = movieCreateValues();
    const number = (id, def, parse) => { const v = parse(document.getElementById(id).value); return isNaN(v) ? def : v; };
    const payload = {
        title,
        destination_path: dest,
        suffix,
        original_language: document.getElementById('movie-create-lang').value.trim() || 'en',
        number_cut: number('movie-create-cut', 10, v => parseInt(v, 10)),
        cut_file_to_get_delay_second_method: number('movie-create-ratio', 2.0, parseFloat)
    };
    // full data: no id is null
    const ids = movieCreateTab.meta.getIds();
    payload.metadata_ids = Object.keys(ids).length ? ids : null;
    const button = document.getElementById('movie-create-btn');
    button.disabled = true;
    try {
        const result = await movieApi.post('movies', payload);
        showToast(`${result.message || 'Movie saved'}: ${result.title} (#${result.movie_id})`, 'success', 5000);
        invalidateMovies();
        const movie = (await fetchMovies(true)).find(m => m.movie_id === result.movie_id) || result;
        if (movieCreateTab.returnGroupId !== null && movie) {
            linkMovieToRegexGroup(movieCreateTab.returnGroupId, movie);
            showToast('Linked to its Bulk Regex group: "← Back to Bulk Regex"', 'info', 5000);
        }
    } catch (e) {
        showToast('Movie not saved: ' + e.message, 'error', 8000);
    } finally {
        await checkMovieCreateTitle();
    }
}

// --- M4b. Bulk Regex tab ---
//
// A board of groups of downloaded files, one movie rule per group. The engine
// groups (groupByNamePattern) and builds the template; the rule has no
// episode group. A file an existing movie rule already matches is listed
// apart under that rule.

let movieRegexTab = null;

function initMovieRegexTab() {
    if (movieRegexTab) { renderMovieRegexBoard(); return; }
    movieRegexTab = { groups: [], nextId: 1, covered: [], rulesByMovie: new Map() };
    document.getElementById('movie-regex-add').onclick = () => openFileModal(addMovieRegexFiles);
    fetchUiConfig().catch(() => null).then(renderMovieRegexBoard);
}

// mode 'new': a new rule for the files; mode 'replace': the regex of an
// existing rule (`rule`) checked on real files of the watched folder.
function newMovieRegexGroup(files) {
    return { id: movieRegexTab.nextId++, mode: 'new', files: [...files], regexManual: null, renameManual: null, movie: null, weight: 1, saved: false, error: null, result: null };
}

function movieRenameOf(group) {
    return group.renameManual !== null ? group.renameManual : movieRenameFor(group.files);
}

// One rename pattern field (input, live preview of the final name with the
// movie's suffix, validation). options: value, files() -> names (the first
// one is the example), movie() -> movie or null, onInput(value),
// repropose() -> value or null (no button). field.problem() -> null or text.
function createMovieRenameField(options) {
    const box = el('div', 'form-group mt-2 movie-rename-box');
    box.appendChild(el('label', 'label', `Rename Pattern (${MOVIE_SUFFIX_PLACEHOLDER} = the movie's suffix; empty = the file keeps its name)`));
    const row = el('div', 'flex items-center movie-inline-row');
    const input = movieInput('text', 'input font-mono pattern-input movie-rename-input', options.value || '');
    input.placeholder = `e.g. Movie A (2001)${MOVIE_SUFFIX_PLACEHOLDER}.mkv`;
    row.appendChild(input);
    if (options.repropose) {
        row.appendChild(movieButton('Re-propose', 'btn btn-secondary btn-sm movie-rename-repropose', () => {
            input.value = options.repropose();
            input.dispatchEvent(new Event('input'));
        }, 'The first name without the parts that differ between the files, {<suffix>} before the extension'));
    }
    const message = el('div', 'mt-1 text-xs validation-msg movie-rename-msg');
    const preview = el('div', 'text-sm movie-rename-preview');
    box.append(row, message, preview);
    const field = { box, input };
    field.problem = () => movieRenameProblem(input.value);
    field.refresh = () => {
        const problem = field.problem();
        const value = input.value;
        message.textContent = problem ? '⚠ Rename pattern ' + problem : (value ? `✓ Holds ${MOVIE_SUFFIX_PLACEHOLDER}` : 'No rename pattern: the file keeps its downloaded name');
        message.className = `mt-1 text-xs validation-msg movie-rename-msg ${problem ? 'text-danger' : (value ? 'text-success' : 'text-muted')}`;
        preview.innerHTML = '';
        const files = options.files();
        const movie = options.movie();
        const example = files[0] || '';
        if (problem) return;
        if (!value && !example) return;
        const name = movieRenamedName(value, movie ? movie.suffix : '', example);
        preview.append(el('span', 'text-muted', 'Final file: '));
        if (movie) preview.append(el('span', 'font-mono text-muted', `${movieRel(movie.destination_path)}/`));
        preview.append(el('span', 'font-mono text-success movie-rename-final', movie || !value ? name : value.split(MOVIE_SUFFIX_PLACEHOLDER).join('<suffix>')));
        if (!movie && value) preview.append(el('span', 'text-muted', ' (link a movie: its suffix replaces <suffix>)'));
        else if (movie) preview.append(el('span', 'text-muted', ` (suffix "${movie.suffix || ''}"${value ? '' : ', name kept'})`));
    };
    input.oninput = () => { if (options.onInput) options.onInput(input.value); field.refresh(); };
    field.refresh();
    return field;
}

function movieBoardGroups() {
    return movieRegexTab.groups.filter(g => g.mode !== 'replace');
}

function movieRegexOf(group) {
    return group.regexManual !== null ? group.regexManual : movieRegexFor(group.files);
}

async function addMovieRegexFiles(names) {
    if (!movieRegexTab) initMovieRegexTab();
    const known = new Set([...movieBoardGroups().flatMap(g => g.files), ...movieRegexTab.covered.map(c => c.name)]);
    const fresh = names.filter(n => !known.has(n));
    if (!fresh.length) return;
    let rules = [];
    try { rules = await fetchAllMovieRules(); } catch (e) { showToast('Movie rules not loaded: ' + e.message, 'error'); }
    const free = [];
    fresh.forEach(name => {
        const rule = rules.find(r => movieRegexMatches(r.regex_pattern, name));
        if (rule) movieRegexTab.covered.push({ name, rule });
        else free.push(name);
    });
    groupByNamePattern(free).forEach(files => movieRegexTab.groups.push(newMovieRegexGroup(files)));
    renderMovieRegexCovered();
    renderMovieRegexBoard();
}

function renderMovieRegexCovered() {
    const box = document.getElementById('movie-regex-covered');
    const content = document.getElementById('movie-regex-covered-content');
    content.innerHTML = '';
    document.getElementById('movie-regex-covered-count').textContent = String(movieRegexTab.covered.length);
    box.classList.toggle('hidden', !movieRegexTab.covered.length);
    const byRule = new Map();
    movieRegexTab.covered.forEach(entry => {
        if (!byRule.has(entry.rule.regex_pattern)) byRule.set(entry.rule.regex_pattern, []);
        byRule.get(entry.rule.regex_pattern).push(entry);
    });
    byRule.forEach(entries => {
        const rule = entries[0].rule;
        const row = el('div', 'rule-row');
        const left = el('div', 'rule-row-patterns');
        left.appendChild(el('div', 'font-mono text-accent mb-1', rule.regex_pattern));
        left.appendChild(el('div', 'text-muted text-sm', `→ ${movieLabel(rule)} — ${movieRel(rule.destination_path)} (#${rule.movie_id})`));
        left.appendChild(el('div', 'text-muted text-xs font-mono movie-rule-rename', rule.rename_pattern ? `rename: ${rule.rename_pattern}` : 'no rename pattern: names kept'));
        const files = el('div', 'rule-files');
        files.appendChild(el('div', 'text-xs text-muted uppercase', `Already matched by this rule (${entries.length})`));
        entries.forEach(entry => {
            const line = el('div', 'rule-file');
            const name = el('span', 'rule-file-name font-mono', entry.name);
            name.title = entry.name;
            line.append(name, movieButton('Propose a new rule anyway', 'btn btn-secondary btn-sm rule-file-propose', () => {
                movieRegexTab.covered = movieRegexTab.covered.filter(c => c !== entry);
                movieRegexTab.groups.push(newMovieRegexGroup([entry.name]));
                renderMovieRegexCovered();
                renderMovieRegexBoard();
            }, 'Put this file on the board below to give it a rule of its own'));
            files.appendChild(line);
        });
        left.appendChild(files);
        const side = el('div', 'movie-rule-side');
        side.append(el('div', 'badge', `W: ${rule.weight}`), movieButton('Edit regex', 'btn btn-secondary btn-sm movie-rule-replace', () => startMovieRegexReplace(rule, entries.map(e => e.name)), 'Change this rule\'s regex, checked on these files and the other files of the watched folder it matches'));
        row.append(left, side);
        content.appendChild(row);
    });
}

function moveMovieRegexFile(name, fromId, toId) {
    if (fromId === toId) return;
    const from = movieRegexTab.groups.find(g => g.id === fromId);
    if (!from || from.mode === 'replace') return;
    if (toId !== null && movieRegexTab.groups.some(g => g.id === toId && g.mode === 'replace')) return;
    from.files = from.files.filter(f => f !== name);
    from.saved = false;
    from.regexManual = null;
    from.renameManual = null;
    if (toId === null) {
        const own = newMovieRegexGroup([name]);
        own.movie = from.movie;
        own.weight = from.weight;
        movieRegexTab.groups.push(own);
    } else {
        const to = movieRegexTab.groups.find(g => g.id === toId);
        if (!to) return;
        to.files.push(name);
        to.saved = false;
        to.regexManual = null;
        to.renameManual = null;
    }
    movieRegexTab.groups = movieRegexTab.groups.filter(g => g.files.length);
    renderMovieRegexBoard();
}

function wireMovieDrop(node, groupId) {
    node.addEventListener('dragover', (e) => { e.preventDefault(); node.classList.add('drag-over'); });
    node.addEventListener('dragleave', () => node.classList.remove('drag-over'));
    node.addEventListener('drop', (e) => {
        e.preventDefault();
        node.classList.remove('drag-over');
        try {
            const data = JSON.parse(e.dataTransfer.getData('text/plain'));
            if (data.board === 'movie') moveMovieRegexFile(data.name, data.groupId, groupId);
        } catch (err) { /* not one of ours */ }
    });
}

function linkMovieToRegexGroup(groupId, movie) {
    if (!movieRegexTab) return;
    const group = movieRegexTab.groups.find(g => g.id === groupId);
    if (!group) return;
    group.movie = movie;
    group.saved = false;
    group.error = null;
    renderMovieRegexBoard();
}

function renderMovieRegexBoard() {
    const container = document.getElementById('movie-regex-cards');
    container.innerHTML = '';
    const replacing = movieRegexTab.groups.filter(g => g.mode === 'replace');
    replacing.forEach(group => container.appendChild(renderMovieReplaceCard(group)));
    const groups = movieBoardGroups();
    if (!groups.length) {
        container.appendChild(el('div', 'text-muted text-sm empty-board', 'Add the downloaded files of one or several movies: "+ Add Files".'));
        return;
    }
    const fileCount = groups.reduce((n, g) => n + g.files.length, 0);
    const bar = el('div', 'board-toolbar');
    bar.appendChild(el('span', 'text-muted text-sm board-summary', `${fileCount} file${fileCount > 1 ? 's' : ''} in ${groups.length} group${groups.length > 1 ? 's' : ''}`));
    const buttons = el('div', 'board-toolbar-buttons');
    const allFiles = () => groups.flatMap(g => g.files);
    buttons.append(
        movieButton('Auto-group', 'btn btn-secondary btn-sm board-auto-group', () => { movieRegexTab.groups = [...replacing, ...groupByNamePattern(allFiles()).map(newMovieRegexGroup)]; renderMovieRegexBoard(); }, 'Regroup every file by the words its name shares with the others (manual edits are dropped)'),
        movieButton('Split all', 'btn btn-secondary btn-sm board-split-all', () => { movieRegexTab.groups = [...replacing, ...allFiles().map(n => newMovieRegexGroup([n]))]; renderMovieRegexBoard(); }, 'One file per card (manual edits are dropped)'),
        movieButton('Clear', 'btn btn-secondary btn-sm', () => { movieRegexTab.groups = replacing; movieRegexTab.covered = []; renderMovieRegexCovered(); renderMovieRegexBoard(); }, 'Remove every file from the board (the "Edit rule" cards stay)')
    );
    bar.appendChild(buttons);
    container.appendChild(bar);
    groups.forEach(group => container.appendChild(renderMovieRegexCard(group)));
    const zone = el('div', 'drop-zone', 'Drop a file here to start a new group');
    wireMovieDrop(zone, null);
    container.appendChild(zone);
}

function renderMovieRegexCard(group) {
    const card = el('div', `card group-card movie-group-card${group.saved ? ' saved' : ''}`);
    card.dataset.groupId = String(group.id);
    wireMovieDrop(card, group.id);

    const header = el('div', 'group-header');
    const title = el('div');
    title.appendChild(el('span', 'text-muted text-xs uppercase font-bold', 'Movie rule'));
    title.appendChild(el('span', 'badge', `${group.files.length} file${group.files.length > 1 ? 's' : ''}`));
    if (group.saved) title.appendChild(el('span', 'badge badge-success', 'saved ✓'));
    header.append(title, movieButton('🗑️', 'btn-icon', () => { movieRegexTab.groups = movieRegexTab.groups.filter(g => g !== group); renderMovieRegexBoard(); }, 'Remove this group'));
    card.appendChild(header);

    const list = el('div', 'group-files');
    group.files.forEach(name => {
        const row = el('div', 'group-file');
        row.draggable = true;
        row.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', JSON.stringify({ board: 'movie', name, groupId: group.id }));
            e.dataTransfer.effectAllowed = 'move';
            row.classList.add('dragging');
        });
        row.addEventListener('dragend', () => row.classList.remove('dragging'));
        row.appendChild(el('span', 'drag-handle', '⠿'));
        const label = el('span', 'group-file-name font-mono', name);
        label.title = name;
        row.append(label, el('span', 'badge'));
        if (group.files.length > 1) row.appendChild(movieButton('−', 'btn-icon eject', () => moveMovieRegexFile(name, group.id, null), 'Take this file out to its own group'));
        list.appendChild(row);
    });
    card.appendChild(list);

    // Regex
    const regexBox = el('div', 'form-group mt-2');
    regexBox.appendChild(el('label', 'label', 'Regex Pattern (Python, no episode group)'));
    const regexRow = el('div', 'flex items-center movie-inline-row');
    const regexInput = movieInput('text', 'input font-mono pattern-input movie-regex-input', movieRegexOf(group));
    regexRow.append(regexInput, movieButton('Re-propose', 'btn btn-secondary btn-sm', () => { group.regexManual = null; group.saved = false; renderMovieRegexBoard(); }));
    const regexMsg = el('div', 'mt-1 text-xs validation-msg');
    regexBox.append(regexRow, regexMsg);
    card.appendChild(regexBox);

    // Rename pattern (proposed from the names, previewed with the movie's suffix)
    const rename = createMovieRenameField({
        value: movieRenameOf(group),
        files: () => group.files,
        movie: () => group.movie,
        onInput: value => { group.renameManual = value; group.saved = false; refreshCheck(); },
        repropose: () => movieRenameFor(group.files)
    });
    card.appendChild(rename.box);

    // Movie + weight
    const linkRow = el('div', 'movie-link-row');
    const movieBox = el('div', 'form-group movie-link-movie');
    movieBox.appendChild(el('label', 'label', 'Movie'));
    const pickerHost = el('div');
    movieBox.append(pickerHost, movieButton('+ Create a movie', 'btn btn-secondary btn-sm mt-1 movie-create-shortcut', () => startMovieCreation(guessMovieTitle(group.files[0]), group.id), 'Opens Movie Creation with a title guessed from the first file; the new movie is linked to this group'));
    const weightBox = el('div', 'form-group');
    weightBox.appendChild(el('label', 'label', 'Weight'));
    const weightInput = movieInput('number', 'input text-center small-number', group.weight);
    weightInput.min = 1;
    weightInput.oninput = () => { group.weight = parseInt(weightInput.value, 10) || 1; group.saved = false; };
    weightBox.appendChild(weightInput);
    linkRow.append(movieBox, weightBox);
    card.appendChild(linkRow);
    const finalLine = el('div', 'text-sm movie-final-line');
    const existing = el('div', 'movie-existing-rules');
    const errorBox = el('div', 'movie-message hidden');
    card.append(finalLine, existing, errorBox);
    if (group.error) movieMessage(errorBox, group.error, 'danger');
    else if (group.saved && group.result) movieMessage(errorBox, group.result, 'success');

    const footer = el('div', 'group-footer');
    const save = movieButton('Save Rule', 'btn btn-primary btn-sm movie-regex-save', () => saveMovieRegexGroup(group, regexInput.value, rename.input.value, save));
    footer.appendChild(save);
    card.appendChild(footer);

    function refreshCheck() {
        const check = checkMovieRegex(regexInput.value, group.files);
        list.querySelectorAll('.group-file').forEach((row, i) => {
            const badge = row.querySelector('.badge');
            badge.textContent = check.perFile[i].match ? 'match' : 'no match';
            badge.className = `badge ${check.perFile[i].match ? 'badge-success' : 'badge-danger'}`;
        });
        if (check.syntaxError) {
            regexMsg.textContent = '⚠ Invalid regex: ' + check.syntaxError;
            regexMsg.className = 'mt-1 text-xs text-danger validation-msg';
        } else if (!check.allMatch) {
            const failing = check.perFile.filter(f => !f.match).length;
            regexMsg.textContent = `⚠ ${failing} file${failing > 1 ? 's' : ''} not matched`;
            regexMsg.className = 'mt-1 text-xs text-warning validation-msg';
        } else if (check.hasGroup) {
            regexMsg.textContent = '⚠ A movie rule needs no episode group';
            regexMsg.className = 'mt-1 text-xs text-warning validation-msg';
        } else {
            regexMsg.textContent = '✓ Matches every file of the group';
            regexMsg.className = 'mt-1 text-xs text-success validation-msg';
        }
        save.disabled = !check.allMatch || !group.movie || !!rename.problem();
    }

    function refreshMovie() {
        finalLine.innerHTML = '';
        existing.innerHTML = '';
        rename.refresh();
        refreshCheck();
        if (!group.movie) {
            finalLine.appendChild(el('span', 'text-muted', 'Link the group to a movie.'));
            return;
        }
        const movie = group.movie;
        if (movie.file_path) finalLine.appendChild(el('span', 'text-muted', `The movie already has ${movieBaseName(movie.file_path)}: a newcomer is merged with it.`));
        loadMovieRulesFor(movie.movie_id).then(rules => {
            if (group.movie !== movie) return;
            existing.innerHTML = '';
            existing.appendChild(el('div', 'text-xs text-muted uppercase mt-2', rules.length ? `Existing rules of this movie (${rules.length})` : 'This movie has no rule yet'));
            const selection = [...movieRegexTab.groups.flatMap(g => g.files), ...movieRegexTab.covered.map(c => c.name)];
            rules.forEach(rule => {
                const row = el('div', 'rule-row');
                const left = el('div', 'rule-row-patterns');
                left.appendChild(el('div', 'font-mono text-accent', rule.regex_pattern));
                left.appendChild(el('div', 'text-muted text-xs font-mono movie-rule-rename', rule.rename_pattern ? `rename: ${rule.rename_pattern}` : 'no rename pattern: names kept'));
                const matched = selection.filter(name => movieRegexMatches(rule.regex_pattern, name));
                if (matched.length) {
                    const files = el('div', 'rule-files');
                    files.appendChild(el('div', 'text-xs text-muted uppercase', `Already matches (${matched.length})`));
                    matched.forEach(name => {
                        const line = el('div', 'rule-file');
                        const label = el('span', 'rule-file-name font-mono', name);
                        label.title = name;
                        line.appendChild(label);
                        files.appendChild(line);
                    });
                    left.appendChild(files);
                }
                const side = el('div', 'movie-rule-side');
                const full = { ...rule, movie_id: movie.movie_id, title: movie.title, suffix: movie.suffix, destination_path: movie.destination_path };
                side.append(el('div', 'badge', `W: ${rule.weight}`), movieButton('Edit regex', 'btn btn-secondary btn-sm movie-rule-replace', () => startMovieRegexReplace(full, matched), 'Change this rule\'s regex in a card of its own, checked on the real files it matches'));
                row.append(left, side);
                existing.appendChild(row);
            });
        }).catch(e => { existing.appendChild(el('div', 'text-danger text-sm', 'Rules not loaded: ' + e.message)); });
    }

    regexInput.oninput = () => { group.regexManual = regexInput.value; group.saved = false; refreshCheck(); };
    const picker = createMoviePicker(pickerHost, (movie) => { group.movie = movie; group.saved = false; group.error = null; movieMessage(errorBox, ''); refreshMovie(); }, () => { group.movie = null; refreshMovie(); });
    if (group.movie) picker.select(group.movie, true);
    refreshMovie();
    return card;
}

function loadMovieRulesFor(movieId) {
    const cache = movieRegexTab.rulesByMovie;
    if (!cache.has(movieId)) cache.set(movieId, fetchMovieRules(movieId).catch(e => { cache.delete(movieId); throw e; }));
    return cache.get(movieId);
}

async function saveMovieRegexGroup(group, pattern, renamePattern, button) {
    if (!group.movie) { showToast('Link the group to a movie first', 'error'); return; }
    if (!checkMovieRegex(pattern, group.files).allMatch) { showToast('The regex must match every file of the group', 'error'); return; }
    const problem = movieRenameProblem(renamePattern);
    if (problem) { showToast('Rename pattern ' + problem, 'error'); return; }
    button.disabled = true;
    try {
        const result = await movieApi.post('regex', { regex_pattern: pattern, movie_id: group.movie.movie_id, rename_pattern: renamePattern || null, weight: group.weight, example_filename: group.files[0] });
        group.saved = true;
        group.error = null;
        group.regexManual = pattern;
        group.renameManual = renamePattern;
        group.result = `${result.message || 'Regex saved'} → ${result.final_file_name}`;
        showToast(group.result, 'success', 5000);
        movieRegexTab.rulesByMovie.delete(group.movie.movie_id);
    } catch (e) {
        group.saved = false;
        group.error = e.message;
        showToast('Rule not saved: ' + e.message, 'error', 8000);
    } finally {
        renderMovieRegexBoard();
    }
}

// Editing the regex of an existing rule happens here, on real files: a card
// in replace mode holds the current regex and the watched-folder files it
// matches (plus `extraFiles`, real names already on the page); more files come
// from the file modal. The new regex must match every file of the card, and
// VMSAM checks it on the first one (regex_replace: the pattern, plus the
// rename pattern only when it was edited here; the movie and the weight
// stay). rule: { regex_pattern, movie_id, title, suffix, destination_path,
// weight, rename_pattern }.
async function startMovieRegexReplace(rule, extraFiles) {
    switchTab('movie-regex-tab');
    const open = movieRegexTab.groups.find(g => g.mode === 'replace' && !g.saved && g.rule.regex_pattern === rule.regex_pattern);
    if (open) {
        const card = document.querySelector(`#movie-regex-cards .movie-group-card[data-group-id="${open.id}"]`);
        if (card) card.scrollIntoView({ block: 'start' });
        return open;
    }
    const group = newMovieRegexGroup(extraFiles || []);
    group.mode = 'replace';
    group.rule = rule;
    group.regexManual = rule.regex_pattern;
    group.renameManual = rule.rename_pattern || '';
    group.movie = { movie_id: rule.movie_id, title: rule.title, suffix: rule.suffix, destination_path: rule.destination_path };
    group.loading = true;
    movieRegexTab.groups.unshift(group);
    renderMovieRegexBoard();
    try {
        // the movie as VMSAM has it now (its suffix for the preview)
        const movie = (await fetchMovies()).find(m => m.movie_id === rule.movie_id);
        if (movie) group.movie = movie;
    } catch (e) { /* the rule's own copy is kept */ }
    try {
        const items = await api.listFiles('', 'files');
        const matched = items.filter(i => !i.is_dir && isVideoName(i.name) && movieRegexMatches(rule.regex_pattern, i.name)).map(i => i.name);
        group.files = [...new Set([...group.files, ...matched])];
    } catch (e) {
        group.error = 'Watched folder not listed: ' + e.message;
    }
    group.loading = false;
    renderMovieRegexBoard();
    const card = document.querySelector(`#movie-regex-cards .movie-group-card[data-group-id="${group.id}"]`);
    if (card) card.scrollIntoView({ block: 'start' });
    return group;
}

function renderMovieReplaceCard(group) {
    const rule = group.rule;
    const card = el('div', `card group-card movie-group-card movie-replace-card${group.saved ? ' saved' : ''}`);
    card.dataset.groupId = String(group.id);
    card.dataset.pattern = rule.regex_pattern;

    const header = el('div', 'group-header');
    const title = el('div');
    title.appendChild(el('span', 'text-muted text-xs uppercase font-bold', 'Edit rule'));
    title.appendChild(el('span', 'badge', `${group.files.length} file${group.files.length === 1 ? '' : 's'}`));
    if (group.saved) title.appendChild(el('span', 'badge badge-success', 'saved ✓'));
    header.append(title, movieButton('🗑️', 'btn-icon', () => { movieRegexTab.groups = movieRegexTab.groups.filter(g => g !== group); renderMovieRegexBoard(); }, 'Close this card (the rule is unchanged unless saved)'));
    card.appendChild(header);

    const about = el('div', 'text-sm movie-replace-about');
    const current = el('div');
    current.append(el('span', 'text-muted', group.saved ? 'Regex: ' : 'Current regex: '), el('span', 'font-mono text-accent movie-replace-current', rule.regex_pattern));
    const owner = el('div', 'text-muted');
    owner.textContent = `Movie #${rule.movie_id} ${movieLabel(rule)} — ${movieRel(rule.destination_path)} · weight ${rule.weight} (the weight, the movie and the rename pattern alone change in the Rules tab)`;
    about.append(current, owner);
    card.appendChild(about);

    const list = el('div', 'group-files');
    if (group.loading) list.appendChild(el('div', 'text-muted text-sm', 'Listing the watched folder…'));
    group.files.forEach(name => {
        const row = el('div', 'group-file');
        const label = el('span', 'group-file-name font-mono', name);
        label.title = name;
        row.append(label, el('span', 'badge'), movieButton('−', 'btn-icon eject movie-replace-remove', () => {
            group.files = group.files.filter(f => f !== name);
            group.saved = false;
            renderMovieRegexBoard();
        }, 'Leave this file out of the check'));
        list.appendChild(row);
    });
    card.appendChild(list);
    const addRow = el('div', 'movie-replace-add');
    addRow.appendChild(movieButton('+ Add files', 'btn btn-secondary btn-sm movie-replace-add-btn', () => openFileModal(names => {
        const fresh = names.filter(n => !group.files.includes(n));
        if (!fresh.length) return;
        group.files.push(...fresh);
        group.saved = false;
        renderMovieRegexBoard();
    }), 'Add files of the watched folder the new regex must match too'));
    card.appendChild(addRow);

    const regexBox = el('div', 'form-group mt-2');
    regexBox.appendChild(el('label', 'label', 'New regex (Python, no episode group)'));
    const regexRow = el('div', 'flex items-center movie-inline-row');
    const regexInput = movieInput('text', 'input font-mono pattern-input movie-regex-input', group.regexManual);
    regexRow.append(regexInput, movieButton('Re-propose', 'btn btn-secondary btn-sm', () => {
        group.regexManual = movieRegexFor(group.files);
        group.saved = false;
        renderMovieRegexBoard();
    }, 'A regex built from the files of this card'), movieButton('Reset', 'btn btn-secondary btn-sm movie-replace-reset', () => {
        group.regexManual = rule.regex_pattern;
        group.saved = false;
        renderMovieRegexBoard();
    }, 'Back to the current regex'));
    const regexMsg = el('div', 'mt-1 text-xs validation-msg');
    regexBox.append(regexRow, regexMsg);
    card.appendChild(regexBox);

    // the rule's rename pattern, kept (sent only when edited here)
    const rename = createMovieRenameField({
        value: group.renameManual,
        files: () => group.files,
        movie: () => group.movie,
        onInput: value => { group.renameManual = value; group.saved = false; refreshCheck(); }
    });
    card.appendChild(rename.box);

    const errorBox = el('div', 'movie-message hidden');
    card.appendChild(errorBox);
    if (group.error) movieMessage(errorBox, group.error, 'danger');
    else if (group.saved && group.result) movieMessage(errorBox, group.result, 'success');

    const footer = el('div', 'group-footer');
    const save = movieButton('Save new regex', 'btn btn-primary btn-sm movie-replace-save', () => saveMovieRegexReplace(group, regexInput.value, rename.input.value, save));
    footer.appendChild(save);
    card.appendChild(footer);

    const refreshCheck = () => {
        const check = checkMovieRegex(regexInput.value, group.files);
        list.querySelectorAll('.group-file').forEach((row, i) => {
            const badge = row.querySelector('.badge');
            badge.textContent = check.perFile[i].match ? 'match' : 'no match';
            badge.className = `badge ${check.perFile[i].match ? 'badge-success' : 'badge-danger'}`;
        });
        let text;
        let kind;
        if (group.loading) { text = ''; kind = 'muted'; }
        else if (!group.files.length) { text = '⚠ No file of the watched folder matches this rule. VMSAM checks the new regex on a real file name: add one with "+ Add files" to save.'; kind = 'warning'; }
        else if (check.syntaxError) { text = '⚠ Invalid regex: ' + check.syntaxError; kind = 'danger'; }
        else if (!check.allMatch) {
            const failing = check.perFile.filter(f => !f.match).length;
            text = `⚠ ${failing} file${failing > 1 ? 's' : ''} not matched`;
            kind = 'warning';
        } else if (check.hasGroup) { text = '⚠ A movie rule needs no episode group'; kind = 'warning'; }
        else if (regexInput.value === rule.regex_pattern && !group.saved && rename.input.value !== (rule.rename_pattern || '')) { text = 'The regex is unchanged: a rename pattern alone is changed in the Rules tab'; kind = 'warning'; }
        else if (regexInput.value === rule.regex_pattern) { text = group.saved ? '✓ Saved' : 'Unchanged: edit the regex'; kind = group.saved ? 'success' : 'muted'; }
        else { text = `✓ Matches every file of the card (VMSAM checks it on ${group.files[0]})`; kind = 'success'; }
        regexMsg.textContent = text;
        regexMsg.className = `mt-1 text-xs text-${kind} validation-msg`;
        save.disabled = group.loading || !group.files.length || !check.allMatch || regexInput.value === rule.regex_pattern || !!rename.problem();
    };
    regexInput.oninput = () => { group.regexManual = regexInput.value; group.saved = false; refreshCheck(); };
    refreshCheck();
    return card;
}

async function saveMovieRegexReplace(group, pattern, renamePattern, button) {
    const old = group.rule.regex_pattern;
    if (!group.files.length) { showToast('Add a real file name first: VMSAM checks the regex on it', 'error'); return; }
    if (!checkMovieRegex(pattern, group.files).allMatch) { showToast('The new regex must match every file of the card', 'error'); return; }
    const problem = movieRenameProblem(renamePattern);
    if (problem) { showToast('Rename pattern ' + problem, 'error'); return; }
    const payload = { regex_pattern: old, new_regex_pattern: pattern, example_filename: group.files[0] };
    const renameEdited = renamePattern !== (group.rule.rename_pattern || '');
    if (renameEdited) payload.rename_pattern = renamePattern || null;
    button.disabled = true;
    try {
        const result = await movieApi.post('regex_replace', payload);
        group.saved = true;
        group.error = null;
        group.regexManual = pattern;
        group.rule = { ...group.rule, regex_pattern: pattern, rename_pattern: renameEdited ? (renamePattern || null) : group.rule.rename_pattern };
        group.result = `${result.message || 'Regex replaced'}${result.final_file_name ? ' → ' + result.final_file_name : ''}`;
        showToast(group.result, 'success', 5000);
        movieRegexTab.rulesByMovie.delete(group.rule.movie_id);
        // files listed under the old rule: kept under the new one when it
        // still matches them, back on the board otherwise
        const lost = [];
        movieRegexTab.covered = movieRegexTab.covered.filter(entry => {
            if (entry.rule.regex_pattern !== old) return true;
            if (movieRegexMatches(pattern, entry.name)) { entry.rule = { ...entry.rule, regex_pattern: pattern }; return true; }
            lost.push(entry.name);
            return false;
        });
        if (lost.length) groupByNamePattern(lost).forEach(files => movieRegexTab.groups.push(newMovieRegexGroup(files)));
        renderMovieRegexCovered();
    } catch (e) {
        group.saved = false;
        group.error = e.message;
        showToast('Regex not replaced: ' + e.message, 'error', 8000);
    } finally {
        renderMovieRegexBoard();
    }
}

// --- M4c. Index tab ---

let movieIndexTab = null;

function initMovieIndexTab() {
    if (movieIndexTab) { if (movieIndexTab.movie) showIndexMovie(movieIndexTab.movie.movie_id); return; }
    movieIndexTab = { movie: null, picker: null, dirPicker: null };
    movieIndexTab.picker = createMoviePicker(document.getElementById('movie-index-picker'), (movie) => showIndexMovie(movie.movie_id), clearIndexMovie);
    movieIndexTab.dirPicker = createMovieFolderPicker(document.getElementById('movie-index-dir-picker'), showIndexDirMovies, () => { document.getElementById('movie-index-dir-movies').innerHTML = ''; });
    document.getElementById('movie-index-browse').onclick = () => openMovieFolderModal('Directory holding the movie', folder => movieIndexTab.dirPicker.select(folder), movieParentDir(movieIndexTab.dirPicker.selected));
    document.getElementById('movie-index-reload').onclick = () => { if (movieIndexTab.movie) showIndexMovie(movieIndexTab.movie.movie_id); };
    document.getElementById('movie-index-run').onclick = runIndexMovie;
    fetchUiConfig().catch(e => showToast('Cannot read the library root: ' + e.message, 'error'));
}

async function showIndexDirMovies(folder) {
    const box = document.getElementById('movie-index-dir-movies');
    box.innerHTML = '';
    try {
        const movies = (await fetchMovies()).filter(m => normalizePath(m.destination_path) === normalizePath(folder.destination_path));
        if (!movies.length) box.appendChild(el('div', 'text-muted text-sm', 'No movie directly in this directory'));
        movies.forEach(movie => {
            const row = el('div', 'dir-item movie-dir-movie');
            row.appendChild(moviePickerRow(movie));
            row.onclick = () => movieIndexTab.picker.select(movie);
            box.appendChild(row);
        });
    } catch (e) { box.appendChild(el('div', 'text-danger text-sm', e.message)); }
}

function clearIndexMovie() {
    movieIndexTab.movie = null;
    document.getElementById('movie-index-work').classList.add('disabled-area');
    ['movie-index-info', 'movie-index-rules', 'movie-index-files'].forEach(id => { document.getElementById(id).innerHTML = ''; });
    document.getElementById('movie-index-report').classList.add('hidden');
}

async function showIndexMovie(movieId) {
    const info = document.getElementById('movie-index-info');
    const rulesBox = document.getElementById('movie-index-rules');
    const filesBox = document.getElementById('movie-index-files');
    document.getElementById('movie-index-work').classList.remove('disabled-area');
    filesBox.innerHTML = '';
    filesBox.appendChild(el('div', 'text-muted text-sm', 'Loading...'));
    let movie, rules, movies, items;
    try {
        await fetchUiConfig();
        movie = await movieApi.get('movie', { movie_id: movieId });
        movieIndexTab.movie = movie;
        [rules, movies] = await Promise.all([fetchMovieRules(movieId), fetchMovies(true)]);
        rules.sort((a, b) => b.weight - a.weight);
    } catch (e) {
        filesBox.innerHTML = '';
        filesBox.appendChild(el('div', 'text-danger text-sm', e.message));
        return;
    }
    if (!movieIndexTab.picker.selected || movieIndexTab.picker.selected.movie_id === movie.movie_id) movieIndexTab.picker.select(movie, true);

    info.innerHTML = '';
    [
        ['Directory', movieRel(movie.destination_path)],
        ['Suffix', movie.suffix ? `"${movie.suffix}"` : '(none)'],
        ['Registered file', movie.file_path ? `${movieBaseName(movie.file_path)} (weight ${movie.file_weight})` : 'none']
    ].forEach(([key, value]) => {
        const line = el('div');
        line.append(el('span', 'text-muted', key + ': '), el('span', 'font-mono text-accent', value));
        info.appendChild(line);
    });

    rulesBox.innerHTML = '';
    rulesBox.appendChild(el('div', 'text-xs text-muted uppercase mt-2', rules.length ? `Rules of this movie (${rules.length}, heaviest first)` : 'This movie has no rule yet: add one from a file below'));
    rules.forEach(rule => {
        const row = el('div', 'rule-row');
        const left = el('div', 'rule-row-patterns');
        left.append(el('div', 'font-mono text-accent', rule.regex_pattern), el('div', 'text-muted text-xs font-mono movie-rule-rename', rule.rename_pattern ? `rename: ${rule.rename_pattern} → ${movieRenamedName(rule.rename_pattern, movie.suffix, '')}` : 'no rename pattern: the name is kept'));
        row.append(left, el('div', 'badge', `W: ${rule.weight}`));
        rulesBox.appendChild(row);
    });

    try {
        items = await api.listFiles(movie.destination_path, 'create');
    } catch (e) {
        filesBox.innerHTML = '';
        filesBox.appendChild(el('div', 'text-danger text-sm', 'Directory not listed: ' + e.message));
        return;
    }
    const files = items.filter(i => !i.is_dir && !i.name.startsWith('.'));
    const registered = new Map(movies.filter(m => m.file_path).map(m => [normalizePath(m.file_path), m]));
    filesBox.innerHTML = '';
    filesBox.appendChild(el('div', 'text-xs text-muted uppercase mt-2', `Files of the directory (${files.length})`));
    if (!files.length) { filesBox.appendChild(el('div', 'text-muted text-sm', 'The directory holds no file')); return; }
    const table = el('table', 'report-table movie-files-table');
    files.forEach(file => {
        const row = el('tr');
        row.dataset.name = file.name;
        const statusCell = el('td', 'movie-file-status');
        const actionCell = el('td', 'movie-actions-cell');
        const owner = registered.get(normalizePath(`${movie.destination_path}/${file.name}`));
        const rule = rules.find(r => movieRegexMatches(r.regex_pattern, file.name));
        if (owner && owner.movie_id === movie.movie_id) statusCell.appendChild(el('span', 'badge badge-success', 'registered file of this movie'));
        else if (owner) statusCell.appendChild(el('span', 'badge', `file of movie #${owner.movie_id} ${movieLabel(owner)}`));
        else if (rule) {
            statusCell.appendChild(el('span', 'badge badge-success', `matched (W ${rule.weight})`));
            if (!isVideoName(file.name)) statusCell.appendChild(el('span', 'badge badge-danger', 'not a video'));
            statusCell.appendChild(el('div', 'font-mono text-xs text-muted', rule.regex_pattern));
            const renamed = movieRenamedName(rule.rename_pattern, movie.suffix, file.name);
            statusCell.appendChild(el('div', 'font-mono text-xs text-success movie-index-new-name', renamed === file.name ? 'name kept' : `→ ${renamed}`));
        } else {
            statusCell.appendChild(el('span', 'badge badge-danger', isVideoName(file.name) ? 'no rule' : 'no rule, not a video'));
        }
        row.append(el('td', 'font-mono', file.name), statusCell, actionCell);
        table.appendChild(row);
        if (!owner && !rule) {
            const formRow = el('tr', 'hidden movie-inline-form-row');
            const formCell = el('td');
            formCell.colSpan = 3;
            formRow.appendChild(formCell);
            actionCell.appendChild(movieButton('+ Rule', 'btn btn-secondary btn-sm movie-index-add-rule', () => {
                formRow.classList.toggle('hidden');
                if (!formCell.children.length) formCell.appendChild(renderInlineMovieRule(movie, file.name, () => showIndexMovie(movie.movie_id)));
            }, 'Add a rule of this movie matching this file'));
            table.appendChild(formRow);
        }
    });
    filesBox.appendChild(table);
}

// Inline form: a rule of `movie` built from one file name, with its rename
// pattern (the name with {<suffix>} before the extension).
function renderInlineMovieRule(movie, fileName, onSaved) {
    const box = el('div', 'movie-inline-rule');
    const input = movieInput('text', 'input font-mono pattern-input movie-inline-pattern', movieRegexFor([fileName]));
    const weight = movieInput('number', 'input text-center small-number', 1);
    weight.min = 1;
    const message = el('div', 'text-xs mt-1');
    const error = el('div', 'movie-message hidden');
    let check = () => {};
    const rename = createMovieRenameField({
        value: movieRenameFor([fileName]),
        files: () => [fileName],
        movie: () => movie,
        onInput: () => check(),
        repropose: () => movieRenameFor([fileName])
    });
    const save = movieButton('Save rule', 'btn btn-primary btn-sm movie-inline-save', async () => {
        save.disabled = true;
        try {
            const result = await movieApi.post('regex', { regex_pattern: input.value, movie_id: movie.movie_id, rename_pattern: rename.input.value || null, weight: parseInt(weight.value, 10) || 1, example_filename: fileName });
            showToast(`${result.message || 'Regex saved'} → ${result.final_file_name}`, 'success', 5000);
            onSaved();
        } catch (e) {
            movieMessage(error, e.message, 'danger');
            showToast('Rule not saved: ' + e.message, 'error', 8000);
            save.disabled = false;
        }
    });
    check = () => {
        const ok = checkMovieRegex(input.value, [fileName]).allMatch;
        message.textContent = ok ? '✓ Matches the file' : '⚠ Does not match the file';
        message.className = `text-xs mt-1 ${ok ? 'text-success' : 'text-warning'}`;
        save.disabled = !ok || !!rename.problem();
    };
    input.oninput = check;
    const row = el('div', 'flex items-center movie-inline-row');
    row.append(input, weight, save);
    box.append(el('label', 'label', `New rule of ${movieLabel(movie)} (pattern, weight)`), row, message, rename.box, error);
    check();
    return box;
}

const MOVIE_SKIP_REASONS = {
    not_a_video_file: 'not a video file',
    movie_already_has_a_file: 'the movie already has a file (nothing is merged here)',
    not_selected: 'another file was taken (heavier rule or larger file)',
    target_exists: 'the final name already exists on disk',
    target_registered: 'the final name is another movie\'s file',
    rename_failed: 'the rename failed',
    invalid_rename_pattern: 'the rule\'s rename pattern gives no valid file name with this suffix'
};

async function runIndexMovie() {
    const movie = movieIndexTab.movie;
    if (!movie) return;
    const button = document.getElementById('movie-index-run');
    const report = document.getElementById('movie-index-report');
    button.disabled = true;
    try {
        const result = await movieApi.post('index_movie', { movie_id: movie.movie_id });
        showToast(result.message || 'Indexed', (result.indexed || []).length ? 'success' : 'info', 5000);
        report.innerHTML = '';
        report.classList.remove('hidden');
        report.appendChild(el('h3', 'movie-report-title', `Indexing report: ${result.message}`));
        if ((result.indexed || []).length) {
            report.appendChild(el('div', 'text-xs text-muted uppercase mt-2', 'Indexed'));
            const table = el('table', 'report-table movie-report-indexed');
            result.indexed.forEach(item => {
                const row = el('tr');
                row.append(el('td', 'font-mono', item.file_name), el('td', 'font-mono text-success', `→ ${item.new_file_name}`), el('td', null, `W ${item.file_weight}`), el('td', 'font-mono text-xs text-muted', item.regex_pattern));
                table.appendChild(row);
            });
            report.appendChild(table);
        }
        if ((result.skipped || []).length) {
            report.appendChild(el('div', 'text-xs text-muted uppercase mt-2', 'Skipped'));
            const table = el('table', 'report-table movie-report-skipped');
            result.skipped.forEach(item => {
                const row = el('tr');
                const details = Object.entries(item).filter(([k]) => k !== 'file_name' && k !== 'reason').map(([k, v]) => `${k}: ${v}`).join(' · ');
                row.append(el('td', 'font-mono', item.file_name), el('td', 'text-warning', `${item.reason}${MOVIE_SKIP_REASONS[item.reason] ? ' — ' + MOVIE_SKIP_REASONS[item.reason] : ''}`), el('td', 'font-mono text-xs text-muted', details));
                table.appendChild(row);
            });
            report.appendChild(table);
        }
        if ((result.unmatched || []).length) {
            report.appendChild(el('div', 'text-xs text-muted uppercase mt-2', `Matched by no rule of the movie (${result.unmatched.length})`));
            const list = el('ul', 'font-mono text-sm movie-plain-list movie-report-unmatched');
            result.unmatched.forEach(name => list.appendChild(el('li', null, name)));
            report.appendChild(list);
        }
        report.appendChild(el('div', 'text-sm text-muted mt-2', `Files already registered to a movie: ${result.already_indexed}`));
        invalidateMovies();
        await showIndexMovie(movie.movie_id);
    } catch (e) {
        showToast('Indexing refused: ' + e.message, 'error', 8000);
    } finally {
        button.disabled = false;
    }
}

// --- M4d. Move Folder tab ---

let movieMoveTab = null;

function initMovieMoveTab() {
    if (movieMoveTab) return;
    const picker = createMovieFolderPicker(document.getElementById('movie-move-picker'), (folder) => { movieMoveTab.shown = showMovieMoveFolder(folder); }, () => {
        movieMoveTab.folder = null;
        movieMoveTab.relative = null;
        document.getElementById('movie-move-current').classList.add('hidden');
        document.getElementById('movie-move-work').classList.add('disabled-area');
        document.getElementById('movie-move-path').value = '';
        updateMovieMovePreview();
    });
    movieMoveTab = { picker, folder: null, relative: null, keepReport: false, shown: null };
    document.getElementById('movie-move-source-browse').onclick = () => openMovieFolderModal('Directory to move', folder => picker.select(folder), movieParentDir(movieMoveTab.folder));
    document.getElementById('movie-move-browse').onclick = openMovieMoveBrowser;
    document.getElementById('movie-move-path').addEventListener('input', updateMovieMovePreview);
    document.getElementById('movie-move-btn').onclick = moveMovieFolder;
    fetchUiConfig().then(config => { document.getElementById('movie-move-root').textContent = config.create_root; })
        .catch(e => showToast('Cannot read the library root: ' + e.message, 'error'));
    updateMovieMovePreview();
}

async function showMovieMoveFolder(folder) {
    movieMoveTab.folder = folder;
    movieMoveTab.relative = null;
    if (!movieMoveTab.keepReport) document.getElementById('movie-move-report').classList.add('hidden');
    document.getElementById('movie-move-current').classList.remove('hidden');
    document.getElementById('movie-move-work').classList.remove('disabled-area');
    const pathNode = document.getElementById('movie-move-current-path');
    const input = document.getElementById('movie-move-path');
    let config;
    try { config = await fetchUiConfig(); } catch (e) { showToast('Cannot read the library root: ' + e.message, 'error'); return; }
    if (movieMoveTab.folder !== folder) return;
    const relative = relativeToRoot(config.create_root, folder.destination_path);
    movieMoveTab.relative = relative;
    if (relative === null) {
        pathNode.textContent = `${folder.destination_path} (outside the library root ${config.create_root}: cannot be moved from here)`;
        input.value = '';
    } else {
        pathNode.textContent = relative || '/ (library root)';
        input.value = relative;
    }
    updateMovieMovePreview();
    const moviesBox = document.getElementById('movie-move-movies');
    moviesBox.innerHTML = '';
    try {
        const own = normalizePath(folder.destination_path);
        const movies = (await fetchMovies(true)).filter(m => normalizePath(m.destination_path) === own || normalizePath(m.destination_path).startsWith(own + '/'));
        if (movieMoveTab.folder !== folder) return;
        moviesBox.appendChild(el('div', 'text-muted', `Movies moved with it (${movies.length}):`));
        const list = el('ul', 'movie-plain-list');
        movies.forEach(m => list.appendChild(el('li', 'font-mono', `#${m.movie_id} ${movieLabel(m)}${normalizePath(m.destination_path) !== own ? ' — in ' + movieRel(m.destination_path) : ''}`)));
        moviesBox.appendChild(list);
    } catch (e) { moviesBox.appendChild(el('div', 'text-danger', e.message)); }
}

function movieMoveName() {
    if (!movieMoveTab.folder || movieMoveTab.relative === null) return '';
    return movieMoveTab.relative.split('/').filter(Boolean).pop() || '';
}

function setMovieMoveInside(dir) {
    if (!movieMoveTab.folder) { showToast('Select the directory to move first', 'error'); return; }
    const name = movieMoveName();
    document.getElementById('movie-move-path').value = dir ? `${dir}/${name}` : name;
    updateMovieMovePreview();
}

// The destination picker: the directory chosen receives the moved one under
// its current name; the directory itself is marked and cannot be its own
// destination.
function openMovieMoveBrowser() {
    const tab = movieMoveTab;
    if (!tab.folder || tab.relative === null) { showToast('Select the directory to move first', 'error'); return; }
    const own = tab.relative;
    const name = movieMoveName();
    const typed = cleanRelativePath(document.getElementById('movie-move-path').value);
    movieDirPicker({
        title: `Move "${name}" into…`,
        hint: 'Select the directory to move it into: it keeps its name, and the path can be edited afterwards.',
        start: typed.split('/').slice(0, -1).join('/') || own.split('/').slice(0, -1).join('/'),
        chooseLabel: 'Move inside this one',
        blocked: dir => (own && (dir === own || dir.startsWith(own + '/'))) ? 'A directory cannot move inside itself' : null,
        decorate: (row, item) => {
            if (item.path !== own) return;
            row.classList.add('dir-registered');
            row.querySelector('.dir-name').appendChild(el('span', 'badge badge-success register-badge', 'this directory'));
        },
        onPick: setMovieMoveInside
    });
}

function updateMovieMovePreview() {
    const preview = document.getElementById('movie-move-preview');
    const button = document.getElementById('movie-move-btn');
    const tab = movieMoveTab;
    button.disabled = true;
    if (!tab || !tab.folder) { preview.textContent = 'Select a directory first'; return; }
    if (tab.relative === null) { preview.textContent = 'This directory is outside the library root'; return; }
    const target = cleanRelativePath(document.getElementById('movie-move-path').value);
    const current = tab.relative || '/';
    if (!target) preview.textContent = `${current} → (enter the new path)`;
    else if (target === tab.relative) preview.textContent = `${current} → ${target} (unchanged)`;
    else if (target.split('/').includes('..')) preview.textContent = `${current} → ${target} (".." is refused)`;
    else { preview.textContent = `${current} → ${target}`; button.disabled = false; }
}

async function moveMovieFolder() {
    const tab = movieMoveTab;
    if (!tab.folder || tab.relative === null) return;
    const newPath = cleanRelativePath(document.getElementById('movie-move-path').value);
    if (!newPath) return;
    const confirmed = await confirmDialog(`${tab.relative || '/'}\n→ ${newPath}`, { title: 'Move this directory on disk?', okText: 'Move' });
    if (!confirmed) return;
    const button = document.getElementById('movie-move-btn');
    const report = document.getElementById('movie-move-report');
    button.disabled = true;
    report.classList.add('hidden');
    try {
        const result = await movieApi.post('folders_move', { destination_path: tab.relative, new_destination_path: newPath });
        invalidateMovies();
        const movies = await fetchMovies(true);
        const moved = (result.movies_updated || []).map(id => movies.find(m => m.movie_id === id)).filter(Boolean);
        showToast(`${result.message || 'Folder moved'}: ${moved.length} movie${moved.length === 1 ? '' : 's'}, ${result.files_updated} file path${result.files_updated === 1 ? '' : 's'} updated`, 'success', 6000);
        report.innerHTML = '';
        report.appendChild(el('div', 'text-muted', 'Moved movies:'));
        const list = el('ul', 'font-mono move-removed-list movie-move-moved');
        moved.forEach(m => list.appendChild(el('li', null, `#${m.movie_id} ${movieLabel(m)} → ${movieRel(m.destination_path)}`)));
        report.appendChild(list);
        const removed = result.removed_directories || [];
        if (removed.length) {
            report.appendChild(el('div', 'text-muted', 'Empty directories deleted after the move:'));
            const removedList = el('ul', 'font-mono move-removed-list movie-move-removed');
            removed.forEach(p => removedList.appendChild(el('li', null, movieRel(p))));
            report.appendChild(removedList);
        }
        report.classList.remove('hidden');
        tab.keepReport = true;
        try {
            tab.picker.select({ destination_path: result.new_destination_path, movies_count: moved.length });
            await tab.shown;
        } finally { tab.keepReport = false; }
    } catch (e) {
        showToast('Move refused: ' + e.message, 'error', 8000);
    } finally {
        updateMovieMovePreview();
    }
}

// --- M4e. Movies tab (list, edit, delete) ---

let movieListTab = null;

function initMovieListTab() {
    if (!movieListTab) {
        movieListTab = { movies: [] };
        document.getElementById('movie-list-search').addEventListener('input', renderMovieList);
        document.getElementById('movie-list-reload').onclick = () => loadMovieList();
    }
    loadMovieList();
}

async function loadMovieList() {
    const box = document.getElementById('movie-list-table');
    try {
        await fetchUiConfig();
        movieListTab.movies = await fetchMovies(true);
    } catch (e) {
        box.innerHTML = '';
        box.appendChild(el('div', 'text-danger text-sm', e.message));
        return;
    }
    renderMovieList();
}

function movieSearchText(movie) {
    return [movie.movie_id, movie.title, movie.suffix, movie.destination_path, movie.file_path || '', ...Object.entries(movie.metadata_ids || {}).map(([k, v]) => `${k} ${v}`)].join(' ').toLowerCase();
}

function renderMovieMetadataChips(ids) {
    const box = el('div', 'movie-chips');
    const entries = Object.entries(ids || {});
    if (!entries.length) box.appendChild(el('span', 'text-muted', '—'));
    entries.forEach(([provider, value]) => box.appendChild(el('span', 'badge movie-chip font-mono', `${provider}: ${value}`)));
    return box;
}

function renderMovieList() {
    const box = document.getElementById('movie-list-table');
    const query = document.getElementById('movie-list-search').value.trim().toLowerCase();
    const movies = movieListTab.movies.filter(m => !query || movieSearchText(m).includes(query));
    document.getElementById('movie-list-count').textContent = query ? `${movies.length} / ${movieListTab.movies.length}` : String(movies.length);
    box.innerHTML = '';
    if (!movies.length) { box.appendChild(el('div', 'text-muted text-sm', movieListTab.movies.length ? 'No movie matches' : 'No movie yet')); return; }
    const table = el('table', 'report-table movie-table');
    const head = el('tr');
    ['#', 'Title', 'Suffix', 'Directory', 'File', 'W', 'Metadata ids', ''].forEach(h => head.appendChild(el('th', null, h)));
    table.appendChild(head);
    movies.forEach(movie => {
        const row = el('tr');
        row.dataset.movieId = String(movie.movie_id);
        const idsCell = el('td');
        idsCell.appendChild(renderMovieMetadataChips(movie.metadata_ids));
        const actions = el('td', 'movie-actions-cell');
        actions.append(
            movieButton('Edit', 'btn btn-secondary btn-sm movie-edit-btn', () => openMovieEdit(movie)),
            movieButton('Delete', 'btn btn-secondary btn-sm movie-delete-btn', () => deleteMovie(movie))
        );
        row.append(
            el('td', 'text-muted', String(movie.movie_id)),
            el('td', 'movie-cell-title', movie.title),
            el('td', 'font-mono', movie.suffix || ''),
            el('td', 'font-mono text-muted', movieRel(movie.destination_path)),
            el('td', 'font-mono', movie.file_path ? movieBaseName(movie.file_path) : '—'),
            el('td', null, movie.file_weight === null || movie.file_weight === undefined ? '' : String(movie.file_weight)),
            idsCell,
            actions
        );
        table.appendChild(row);
    });
    box.appendChild(table);
}

// Edit. The title is the movie's key (`movies` creates or updates by title):
// read-only here, renaming a movie = delete it and create it again. The
// other fields are posted to `movies` with the title (database only). A new
// directory for a movie that has a file is a move first (`movie_move`,
// confirmed: VMSAM moves the file, same name, and deletes the old
// directories left empty), then the other changes through `movies`. The
// directory is relative to CREATE_ROOT (the relay prefixes it), typed or
// chosen in the directory picker. Shown: the current file and, per rule, the
// name its rename pattern gives with the suffix being edited (a file already
// in place is not renamed: it applies to the next integration or merge).
function openMovieEdit(movie, notice) {
    const card = document.getElementById('movie-edit');
    card.innerHTML = '';
    card.classList.remove('hidden');
    card.dataset.movieId = String(movie.movie_id);
    const header = el('div', 'flex flex-between items-center');
    header.append(el('h2', 'section-title', `Edit movie #${movie.movie_id}`), movieButton('✕', 'btn-icon', () => card.classList.add('hidden'), 'Close'));
    card.appendChild(header);
    const notes = el('div', 'movie-edit-notice movie-message hidden');
    card.appendChild(notes);
    if (notice) {
        movieMessage(notes, notice, 'success');
        notes.classList.add('movie-edit-notice');
    }

    const titleBox = el('div', 'form-group movie-edit-title-box');
    titleBox.appendChild(el('label', 'label', 'Title (name in VMSAM, the key of the movie)'));
    titleBox.appendChild(el('div', 'movie-edit-title text-main', movie.title));
    titleBox.appendChild(el('div', 'text-muted text-xs mt-1', 'Read-only: the title identifies the movie (it is not the file name). Renaming a movie = delete it and create it again under the new title (its rules must be deleted first).'));

    const root = uiConfigCache ? uiConfigCache.create_root : null;
    const currentDir = root ? relativeToRoot(root, movie.destination_path) : null;
    const dest = movieInput('text', 'input font-mono movie-edit-dest', currentDir === null ? '' : currentDir);
    dest.placeholder = 'Directory, relative to the library root';
    const browse = movieButton('Browse…', 'btn btn-secondary movie-edit-browse', () => movieDirPicker({
        title: `Directory of ${movieLabel(movie)}`,
        hint: movie.file_path ? 'Select the directory the movie goes in. On save, VMSAM moves its file there (same name) and deletes the old directories left empty; a directory that does not exist yet can be typed: VMSAM creates it.' : 'Select the directory the movie goes in (no file registered yet: only the database changes).',
        start: dest.value,
        chooseLabel: 'Use this directory',
        onPick: path => { dest.value = path; updatePreview(); }
    }), 'Choose the directory in the library');
    const destRow = el('div', 'flex items-center movie-inline-row');
    destRow.append(dest, browse);
    const destBox = movieLabelled('Directory (relative to the library root)', destRow);
    if (currentDir === null) destBox.appendChild(el('div', 'text-warning text-xs mt-1', `Stored outside the library root (${movie.destination_path}): choose a directory under the root to save.`));

    const suffix = movieInput('text', 'input font-mono movie-edit-suffix', movie.suffix || '');
    const lang = movieInput('text', 'input movie-edit-lang', movie.original_language);
    const cut = movieInput('number', 'input movie-edit-cut', movie.number_cut);
    const ratio = movieInput('number', 'input movie-edit-ratio', movie.cut_file_to_get_delay_second_method);
    ratio.step = '0.1';
    const suffixBox = movieLabelled(`Suffix (replaces ${MOVIE_SUFFIX_PLACEHOLDER} in the rename patterns)`, suffix);
    const grid3 = el('div', 'grid-3');
    grid3.append(movieLabelled('Original Language', lang), movieLabelled('Number Cut', cut), movieLabelled('Cut Ratio (1/N)', ratio));
    const metaHost = el('div', 'movie-edit-meta');
    const preview = el('div', 'text-sm mt-2 movie-edit-preview');
    const rulesBox = el('div', 'text-sm mt-1 movie-edit-rules');
    const report = el('div', 'hidden text-sm mt-2 movie-edit-removed');
    const error = el('div', 'movie-message hidden movie-edit-error');
    card.append(titleBox, destBox, suffixBox, grid3, el('div', 'label mt-2', 'Metadata ids'), metaHost, preview, rulesBox, report, error);

    let rules = null;
    let save = null;
    const updatePreview = () => {
        const dir = cleanRelativePath(dest.value);
        const moved = currentDir === null || dir !== currentDir;
        const problems = [];
        if (/\//.test(suffix.value)) problems.push('the suffix cannot hold a / (it goes in a file name)');
        if (dir.split('/').includes('..')) problems.push('".." is refused in the directory');
        if (currentDir === null && !dir) problems.push('choose a directory under the library root');
        preview.innerHTML = '';
        const shownDir = d => (d || '(library root)') + '/';
        const fileLine = el('div', 'movie-edit-from');
        if (movie.file_path) fileLine.append(el('span', 'text-muted', 'Current file: '), el('span', 'font-mono text-muted', shownDir(currentDir === null ? movie.destination_path : currentDir)), el('span', 'font-mono movie-edit-file', movieBaseName(movie.file_path)));
        else fileLine.append(el('span', 'text-muted', 'Current file: '), el('span', 'text-muted movie-edit-file', 'none registered yet'));
        preview.appendChild(fileLine);
        if (moved) {
            const toLine = el('div', 'movie-edit-to');
            toLine.append(el('span', 'text-muted', '→ '), el('span', 'font-mono text-success', shownDir(dir)), el('span', 'text-muted', movie.file_path ? ` (the file is moved there, same name: ${movieBaseName(movie.file_path)})` : ' (database only: no file yet)'));
            preview.appendChild(toLine);
        }
        rulesBox.innerHTML = '';
        if (rules === null) rulesBox.appendChild(el('div', 'text-muted text-xs', 'Loading the rules…'));
        else if (!rules.length) rulesBox.appendChild(el('div', 'text-muted text-xs', 'No rule yet: the names its files will take come from the rules\' rename patterns (Bulk Regex).'));
        else {
            rulesBox.appendChild(el('div', 'text-xs text-muted uppercase mt-1', `Names given by its rules with this suffix (${rules.length})`));
            const list = el('ul', 'movie-plain-list movie-edit-rule-names');
            rules.forEach(rule => {
                const item = el('li', 'font-mono text-xs');
                item.dataset.pattern = rule.regex_pattern;
                item.append(el('span', 'text-accent', rule.regex_pattern), el('span', 'text-muted', ' → '));
                if (rule.rename_pattern) item.append(el('span', 'text-success movie-edit-rule-name', movieRenamedName(rule.rename_pattern, suffix.value, '')));
                else item.append(el('span', 'text-muted movie-edit-rule-name', 'name kept (no rename pattern)'));
                list.appendChild(item);
            });
            rulesBox.appendChild(list);
            if (movie.file_path && suffix.value !== (movie.suffix || '')) rulesBox.appendChild(el('div', 'text-warning text-xs', 'A new suffix does not rename the file already in place: it applies to the next integration or merge.'));
        }
        if (problems.length) preview.appendChild(el('div', 'text-warning text-xs mt-1', '⚠ ' + problems.join('; ')));
        if (save) save.disabled = problems.length > 0;
    };
    fetchMovieRules(movie.movie_id).then(found => { rules = found; updatePreview(); }).catch(e => { rules = []; rulesBox.appendChild(el('div', 'text-danger text-xs', 'Rules not loaded: ' + e.message)); });
    const meta = createMovieMetadataGenerator(metaHost, [
        { text: 'Use as suffix', use: tags => { suffix.value = ' ' + tags; updatePreview(); } },
        { text: 'Append to suffix', use: tags => { suffix.value = appendMovieTags(suffix.value, tags); updatePreview(); } },
        { text: 'Append to the directory name', title: 'Adds the tags at the end of the last directory of the destination', use: tags => { dest.value = appendMovieTags(cleanRelativePath(dest.value), tags).trim(); updatePreview(); } }
    ]);
    meta.setIds(movie.metadata_ids || {});
    [dest, suffix].forEach(input => input.addEventListener('input', updatePreview));

    save = movieButton('Save changes', 'btn btn-primary movie-edit-save', async () => {
        const dir = cleanRelativePath(dest.value);
        const number = (input, current, parse) => { const v = parse(input.value); return isNaN(v) ? current : v; };
        const ids = meta.getIds();
        const payload = {
            title: movie.title,
            destination_path: dir,
            suffix: suffix.value,
            original_language: lang.value.trim() || movie.original_language,
            number_cut: number(cut, movie.number_cut, v => parseInt(v, 10)),
            cut_file_to_get_delay_second_method: number(ratio, movie.cut_file_to_get_delay_second_method, parseFloat),
            metadata_ids: Object.keys(ids).length ? ids : null
        };
        const sortedJson = obj => JSON.stringify(Object.keys(obj || {}).sort().map(k => [k, obj[k]]));
        const moved = currentDir === null || dir !== currentDir;
        const othersChanged = payload.suffix !== (movie.suffix || '')
            || payload.original_language !== movie.original_language || payload.number_cut !== movie.number_cut
            || payload.cut_file_to_get_delay_second_method !== movie.cut_file_to_get_delay_second_method
            || sortedJson(ids) !== sortedJson(movie.metadata_ids);
        if (!moved && !othersChanged) { showToast('Nothing changed', 'info'); return; }
        const moveFile = moved && !!movie.file_path;
        if (moveFile) {
            const confirmed = await confirmDialog(`${movieBaseName(movie.file_path)}\n${currentDir === null ? movie.destination_path : (currentDir || '(library root)')}\n→ ${dir || '(library root)'}${othersChanged ? '\nThen the other changes are saved.' : ''}`, { title: 'Move this movie\'s file on disk?', okText: 'Move and save' });
            if (!confirmed) return;
        }
        save.disabled = true;
        movieMessage(error, '');
        const parts = [];
        let current = movie;
        try {
            if (moveFile) {
                const result = await movieApi.post('movie_move', { movie_id: movie.movie_id, new_destination_path: dir });
                current = result;
                parts.push(result.message || 'Movie moved');
                if (result.file_moved) parts.push(`file moved to ${movieRel(result.destination_path)}`);
                showRemovedDirectories(report, result.removed_directories, root);
            }
            if (othersChanged || (moved && !moveFile)) {
                const result = await movieApi.post('movies', payload);
                current = result;
                parts.push(result.message || 'Movie updated');
            }
            const text = parts.join(' — ');
            showToast(text, 'success', 6000);
            invalidateMovies();
            await loadMovieList();
            const removed = report.classList.contains('hidden') ? null : report;
            openMovieEdit(current, text);
            if (removed) document.getElementById('movie-edit').querySelector('.movie-edit-removed').replaceWith(removed);
        } catch (e) {
            showToast('Not saved: ' + e.message, 'error', 8000);
            if (!parts.length) {
                movieMessage(error, e.message, 'danger');
                save.disabled = false;
                return;
            }
            // the move went through, the other changes did not: the card
            // reopens on the moved movie, with VMSAM's refusal
            invalidateMovies();
            await loadMovieList();
            openMovieEdit(current, parts.join(' — '));
            movieMessage(document.getElementById('movie-edit').querySelector('.movie-edit-error'), 'Other changes not saved: ' + e.message, 'danger');
        }
    });
    const footer = el('div', 'movie-actions mt-2');
    footer.append(movieButton('Cancel', 'btn btn-secondary', () => card.classList.add('hidden')), save);
    card.appendChild(footer);
    updatePreview();
    card.scrollIntoView({ block: 'start' });
}

async function deleteMovie(movie) {
    const confirmed = await confirmDialog(`#${movie.movie_id} ${movieLabel(movie)}\n${movieRel(movie.destination_path)}${movie.file_path ? '\nIts file stays on disk: ' + movieBaseName(movie.file_path) : ''}`, { title: 'Delete this movie?', okText: 'Delete' });
    if (!confirmed) return;
    try {
        const result = await movieApi.post('movies_delete', { movie_id: movie.movie_id });
        showToast(result.message || 'Movie deleted', 'success', 6000);
        invalidateMovies();
        const card = document.getElementById('movie-edit');
        if (card.dataset.movieId === String(movie.movie_id)) card.classList.add('hidden');
        loadMovieList();
    } catch (e) {
        showToast('Not deleted: ' + e.message, 'error', 10000);
    }
}

// --- M4f. Rules tab ---

let movieRulesTab = null;

function initMovieRulesTab() {
    if (!movieRulesTab) {
        movieRulesTab = { rules: [], editing: null };
        document.getElementById('movie-rules-search').addEventListener('input', renderMovieRules);
        document.getElementById('movie-rules-test').addEventListener('input', renderMovieRules);
        document.getElementById('movie-rules-reload').onclick = () => loadMovieRules();
    }
    loadMovieRules();
}

async function loadMovieRules() {
    const box = document.getElementById('movie-rules-table');
    try {
        await fetchUiConfig();
        movieRulesTab.rules = await fetchAllMovieRules();
    } catch (e) {
        box.innerHTML = '';
        box.appendChild(el('div', 'text-danger text-sm', e.message));
        return;
    }
    renderMovieRules();
}

function renderMovieRules() {
    const box = document.getElementById('movie-rules-table');
    const query = document.getElementById('movie-rules-search').value.trim().toLowerCase();
    const testName = document.getElementById('movie-rules-test').value;
    const all = movieRulesTab.rules;
    const rules = all.filter(r => !query || [r.regex_pattern, r.title, r.suffix, r.rename_pattern || '', r.destination_path, '#' + r.movie_id].join(' ').toLowerCase().includes(query));
    document.getElementById('movie-rules-count').textContent = query ? `${rules.length} / ${all.length}` : String(all.length);
    const testResult = document.getElementById('movie-rules-test-result');
    testResult.innerHTML = '';
    if (testName) {
        const matching = all.filter(r => movieRegexMatches(r.regex_pattern, testName));
        if (!matching.length) testResult.appendChild(el('span', 'text-muted', 'No movie rule matches this name'));
        else testResult.appendChild(el('span', matching.length > 1 ? 'text-warning' : 'text-success', `${matching.length} rule${matching.length > 1 ? 's match' : ' matches'}: ${matching.map(r => '#' + r.movie_id + ' ' + movieLabel(r)).join(', ')}`));
    }
    box.innerHTML = '';
    if (!rules.length) { box.appendChild(el('div', 'text-muted text-sm', all.length ? 'No rule matches the filter' : 'No movie rule yet')); return; }
    const table = el('table', 'report-table movie-table');
    const head = el('tr');
    ['Pattern', 'Movie (title)', 'Rename pattern', 'Directory', 'W', ''].forEach(h => head.appendChild(el('th', null, h)));
    table.appendChild(head);
    rules.forEach(rule => {
        const row = el('tr', testName && movieRegexMatches(rule.regex_pattern, testName) ? 'movie-row-hit' : null);
        row.dataset.pattern = rule.regex_pattern;
        const actions = el('td', 'movie-actions-cell');
        actions.append(
            movieButton('Edit', 'btn btn-secondary btn-sm movie-rule-edit', () => { movieRulesTab.editing = movieRulesTab.editing === rule.regex_pattern ? null : rule.regex_pattern; renderMovieRules(); }, 'Change the weight, the movie or the rename pattern of this rule'),
            movieButton('Edit regex in Bulk Regex', 'btn btn-secondary btn-sm movie-rule-to-bulk', () => startMovieRegexReplace(rule, []), 'The regex is changed in Bulk Regex, checked on the real files of the watched folder'),
            movieButton('Delete', 'btn btn-secondary btn-sm movie-rule-delete', () => deleteMovieRule(rule))
        );
        row.append(
            el('td', 'font-mono text-accent', rule.regex_pattern),
            el('td', 'movie-cell-title', `${movieLabel(rule)} (#${rule.movie_id})`),
            el('td', rule.rename_pattern ? 'font-mono movie-rule-rename-cell' : 'text-muted movie-rule-rename-cell', rule.rename_pattern || 'none: name kept'),
            el('td', 'font-mono text-muted', movieRel(rule.destination_path)),
            el('td', null, String(rule.weight)),
            actions
        );
        table.appendChild(row);
        if (movieRulesTab.editing === rule.regex_pattern) {
            const formRow = el('tr', 'movie-inline-form-row');
            const cell = el('td');
            cell.colSpan = 6;
            cell.appendChild(renderMovieRuleEditor(rule, testName));
            formRow.appendChild(cell);
            table.appendChild(formRow);
        }
    });
    box.appendChild(table);
}

// Weight, movie and rename pattern of a rule, posted to `regex` with full
// data (the pattern unchanged); VMSAM checks the rule on an example file
// name: one built from the regex (movieSampleFromRegex), the tested name when
// the rule matches it, or a real name typed in. The regex itself is edited in
// Bulk Regex. A new rename pattern does not rename a file already in place.
function renderMovieRuleEditor(rule, testName) {
    const box = el('div', 'movie-inline-rule movie-rule-editor');
    const pattern = el('div', 'font-mono text-accent movie-rule-pattern-ro', rule.regex_pattern);
    const toBulk = movieButton('Edit regex in Bulk Regex', 'btn btn-secondary btn-sm movie-rule-to-bulk', () => startMovieRegexReplace(rule, []), 'The regex is changed in Bulk Regex, on the real files it matches');
    const patternRow = el('div', 'flex items-center movie-inline-row movie-rule-pattern-row');
    patternRow.append(pattern, toBulk);
    const weight = movieInput('number', 'input text-center small-number movie-rule-weight', rule.weight);
    weight.min = 1;
    const tested = testName && movieRegexMatches(rule.regex_pattern, testName) ? testName : '';
    const example = movieInput('text', 'input font-mono movie-rule-example', tested || movieSampleFromRegex(rule.regex_pattern) || '');
    example.placeholder = 'A file name the rule matches (VMSAM checks the rule on it)';
    const check = el('div', 'text-xs mt-1');
    const error = el('div', 'movie-message hidden');
    let target = { movie_id: rule.movie_id, title: rule.title, suffix: rule.suffix, destination_path: rule.destination_path };
    const pickerHost = el('div');
    let refresh = () => {};
    const rename = createMovieRenameField({
        value: rule.rename_pattern || '',
        files: () => (example.value ? [example.value] : []),
        movie: () => target,
        onInput: () => refresh()
    });
    const picker = createMoviePicker(pickerHost, (movie) => { target = movie; rename.refresh(); refresh(); }, () => { target = null; rename.refresh(); refresh(); });
    picker.select(target, true);
    // the linked movie as VMSAM has it now (its suffix for the preview)
    fetchMovies().then(movies => {
        const found = movies.find(m => m.movie_id === rule.movie_id);
        if (found && target && target.movie_id === found.movie_id) { target = found; rename.refresh(); }
    }).catch(() => null);
    const save = movieButton('Save', 'btn btn-primary btn-sm movie-rule-save', async () => {
        if (!target) { showToast('Choose the movie', 'error'); return; }
        const problem = rename.problem();
        if (problem) { showToast('Rename pattern ' + problem, 'error'); return; }
        const w = parseInt(weight.value, 10);
        const renamePattern = rename.input.value || null;
        const payload = { regex_pattern: rule.regex_pattern, movie_id: target.movie_id, rename_pattern: renamePattern, weight: isNaN(w) ? rule.weight : w, example_filename: example.value };
        if (payload.movie_id === rule.movie_id && payload.weight === rule.weight && renamePattern === (rule.rename_pattern || null)) { showToast('Nothing changed', 'info'); return; }
        if (payload.movie_id !== rule.movie_id) {
            const confirmed = await confirmDialog(`${rule.regex_pattern}\n#${rule.movie_id} ${movieLabel(rule)}\n→ #${target.movie_id} ${movieLabel(target)}`, { title: 'Link this rule to another movie?', okText: 'Relink' });
            if (!confirmed) return;
        }
        save.disabled = true;
        try {
            const result = await movieApi.post('regex', payload);
            showToast(`${result.message || 'Regex updated'}${result.final_file_name ? ' → ' + result.final_file_name : ''}`, 'success', 5000);
            movieRulesTab.editing = null;
            if (movieRegexTab) movieRegexTab.rulesByMovie.clear();
            loadMovieRules();
        } catch (e) {
            movieMessage(error, e.message, 'danger');
            showToast('Not saved: ' + e.message, 'error', 8000);
            save.disabled = false;
        }
    });
    refresh = () => {
        const name = example.value;
        if (!name) { check.textContent = 'No example could be built from this regex: type a real file name it matches'; check.className = 'text-xs mt-1 text-warning'; }
        else if (movieRegexMatches(rule.regex_pattern, name)) { check.textContent = '✓ The rule matches the example'; check.className = 'text-xs mt-1 text-success'; }
        else { check.textContent = '⚠ The rule does not match this example'; check.className = 'text-xs mt-1 text-warning'; }
        save.disabled = !(name && movieRegexMatches(rule.regex_pattern, name) && target) || !!rename.problem();
    };
    example.oninput = () => { rename.refresh(); refresh(); };
    const fields = el('div', 'movie-rule-fields');
    fields.append(movieLabelled('Movie', pickerHost), movieLabelled('Weight', weight));
    const footer = el('div', 'movie-actions');
    footer.append(movieButton('Cancel', 'btn btn-secondary btn-sm', () => { movieRulesTab.editing = null; renderMovieRules(); }), save);
    box.append(movieLabelled('Regex (read-only here)', patternRow), fields, rename.box, movieLabelled('Example file name', example), check, error, footer);
    refresh();
    return box;
}

async function deleteMovieRule(rule) {
    const confirmed = await confirmDialog(`${rule.regex_pattern}\n#${rule.movie_id} ${movieLabel(rule)}`, { title: 'Delete this movie rule?', okText: 'Delete' });
    if (!confirmed) return;
    try {
        const result = await movieApi.post('regex_delete', { regex_pattern: rule.regex_pattern });
        showToast(result.message || 'Regex deleted', 'success');
        if (movieRegexTab) movieRegexTab.rulesByMovie.delete(rule.movie_id);
        loadMovieRules();
    } catch (e) {
        showToast('Not deleted: ' + e.message, 'error', 8000);
    }
}

// --- M4g. Errors tab ---

let movieErrorsTab = null;
const MOVIE_FUSION_POLL_MS = 5000;

function initMovieErrorsTab() {
    if (!movieErrorsTab) {
        movieErrorsTab = { timer: null, errors: [], movies: [], collapsed: new Set() };
        document.getElementById('movie-errors-reload').onclick = () => loadMovieErrors();
        document.getElementById('movie-errors-search').addEventListener('input', renderMovieErrors);
        document.getElementById('movie-errors-expand').onclick = () => setMovieErrorGroups(false);
        document.getElementById('movie-errors-collapse').onclick = () => setMovieErrorGroups(true);
    }
    // everything listed and expanded on arrival
    movieErrorsTab.collapsed.clear();
    loadMovieErrors();
    pollMovieFusion();
    if (!movieErrorsTab.timer) movieErrorsTab.timer = setInterval(pollMovieFusion, MOVIE_FUSION_POLL_MS);
}

// Polls only while the tab is shown; the timer stops itself otherwise.
async function pollMovieFusion() {
    if (!movieTabShown('movie-errors-tab')) {
        clearInterval(movieErrorsTab.timer);
        movieErrorsTab.timer = null;
        return;
    }
    const box = document.getElementById('movie-fusion-status');
    try {
        const status = await movieApi.get('fusion_status');
        box.innerHTML = '';
        const line = el('div');
        line.append(el('span', 'text-muted', 'Status: '), el('span', status.status === 'idle' ? 'badge' : 'badge badge-success', String(status.status)), el('span', 'text-muted', `  waiting: ${status.queue_length}`));
        box.appendChild(line);
        if (status.current_job) {
            const job = el('div', 'mt-1');
            job.append(el('span', 'text-muted', 'Running: '), el('span', 'font-mono', Object.entries(status.current_job).map(([k, v]) => `${k}: ${v}`).join(' · ')));
            box.appendChild(job);
        }
        if ((status.pending_jobs || []).length) {
            const list = el('ol', 'font-mono movie-plain-list');
            status.pending_jobs.forEach(j => list.appendChild(el('li', null, j.error_file_path)));
            box.appendChild(list);
        }
    } catch (e) {
        box.innerHTML = '';
        box.appendChild(el('div', 'text-danger', 'Fusion status unavailable: ' + e.message));
    }
    movieErrorsTab.polls = (movieErrorsTab.polls || 0) + 1;
}

async function loadMovieErrors() {
    const box = document.getElementById('movie-errors-list');
    try {
        await fetchUiConfig();
        const response = await movieApi.get('errors');
        movieErrorsTab.errors = (response && response.incompatible_files) || [];
        movieErrorsTab.movies = await fetchMovies(true);
    } catch (e) {
        box.innerHTML = '';
        box.appendChild(el('div', 'text-danger text-sm', e.message));
        return;
    }
    renderMovieErrors();
}

// Collapses (true) or expands (false) every group, the hidden ones too.
function setMovieErrorGroups(collapsed) {
    movieErrorsTab.collapsed = new Set(collapsed ? movieErrorsTab.errors.map(item => item.movie_id) : []);
    renderMovieErrors();
}

// One collapsible group per movie. The search keeps a group whose title,
// suffix or directory contains the text (all its files), or the files whose
// path contains it; case-insensitive, no regex.
function renderMovieErrors() {
    const box = document.getElementById('movie-errors-list');
    const { errors, movies, collapsed } = movieErrorsTab;
    const query = document.getElementById('movie-errors-search').value.trim().toLowerCase();
    const byMovie = new Map();
    errors.forEach(item => {
        if (!byMovie.has(item.movie_id)) byMovie.set(item.movie_id, []);
        byMovie.get(item.movie_id).push(item);
    });
    box.innerHTML = '';
    let shownFiles = 0;
    byMovie.forEach((all, movieId) => {
        const first = all[0];
        const movie = movies.find(m => m.movie_id === movieId);
        const groupText = [movieLabel(first), first.destination_path, movieRel(first.destination_path), '#' + movieId].join(' ').toLowerCase();
        const items = !query || groupText.includes(query) ? all : all.filter(item => item.file_path.toLowerCase().includes(query));
        if (!items.length) return;
        shownFiles += items.length;
        const isCollapsed = collapsed.has(movieId);
        const group = el('div', `card movie-error-group${isCollapsed ? ' collapsed' : ''}`);
        group.dataset.movieId = String(movieId);
        const header = el('div', 'flex flex-between items-center movie-error-header');
        header.setAttribute('role', 'button');
        header.tabIndex = 0;
        header.setAttribute('aria-expanded', String(!isCollapsed));
        const titleBox = el('div', 'movie-error-heading');
        titleBox.append(el('span', 'movie-error-icon', isCollapsed ? '▶' : '▼'), el('h3', 'movie-error-title', `#${movieId} ${movieLabel(first)}`));
        header.append(titleBox, el('span', 'badge', items.length === all.length ? `${all.length} file${all.length > 1 ? 's' : ''}` : `${items.length} / ${all.length} files`));
        const toggle = () => {
            if (collapsed.has(movieId)) collapsed.delete(movieId); else collapsed.add(movieId);
            renderMovieErrors();
        };
        header.onclick = toggle;
        header.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } };
        group.append(header, el('div', 'text-sm text-muted font-mono', movieRel(first.destination_path)));
        if (!isCollapsed) {
            const body = el('div', 'movie-error-body');
            body.appendChild(el('div', 'text-sm mb-2', movie && movie.file_path ? `Registered file: ${movieBaseName(movie.file_path)} (weight ${movie.file_weight})` : 'No registered file: a fusion has nothing to merge with'));
            const table = el('table', 'report-table');
            items.forEach(item => {
                const row = el('tr');
                row.dataset.path = item.file_path;
                const actions = el('td', 'movie-actions-cell');
                actions.append(
                    movieButton('Relaunch fusion', 'btn btn-secondary btn-sm movie-error-fusion', () => relaunchMovieFusion(item)),
                    movieButton('Delete', 'btn btn-secondary btn-sm movie-error-delete', () => deleteMovieError(item))
                );
                row.append(el('td', 'font-mono', item.file_path), el('td', null, `W ${item.file_weight}`), actions);
                table.appendChild(row);
            });
            body.appendChild(table);
            group.appendChild(body);
        }
        box.appendChild(group);
    });
    document.getElementById('movie-errors-count').textContent = query ? `${shownFiles} / ${errors.length}` : String(errors.length);
    if (!errors.length) box.appendChild(el('div', 'text-muted text-sm', 'No incompatible file'));
    else if (!box.children.length) box.appendChild(el('div', 'text-muted text-sm', 'Nothing matches the search'));
}

async function relaunchMovieFusion(item) {
    try {
        const result = await movieApi.post('fusion', { error_file_path: item.file_path });
        showToast(`${result.message || 'Fusion queued'}${result.queue_position ? ' (position ' + result.queue_position + ')' : ''}`, 'success', 5000);
        pollMovieFusion();
    } catch (e) {
        showToast('Fusion refused: ' + e.message, 'error', 8000);
    }
}

async function deleteMovieError(item) {
    const confirmed = await confirmDialog(`${item.file_path}\nThe file, its .log.error and the row are deleted, then its error directory if only .error files are left in it.`, { title: 'Delete this incompatible file?', okText: 'Delete' });
    if (!confirmed) return;
    try {
        const result = await movieApi.post('errors_delete', { error_file_path: item.file_path });
        const parts = [`${(result.deleted_files || []).length} file(s) deleted`];
        if (result.file_was_missing) parts.push('the file was already missing');
        if ((result.removed_directories || []).length) parts.push(`directories removed: ${result.removed_directories.map(movieRel).join(', ')}`);
        showToast(`${result.message || 'Error deleted'}: ${parts.join('; ')}`, 'success', 7000);
        (result.warnings || []).forEach(w => showToast(w, 'error', 8000));
        loadMovieErrors();
    } catch (e) {
        showToast('Not deleted: ' + e.message, 'error', 8000);
    }
}

// --- Registration ---

Object.assign(TAB_INIT, {
    'movie-create-tab': () => initMovieCreateTab(),
    'movie-regex-tab': () => initMovieRegexTab(),
    'movie-index-tab': () => initMovieIndexTab(),
    'movie-move-tab': () => initMovieMoveTab(),
    'movie-list-tab': () => initMovieListTab(),
    'movie-rules-tab': () => initMovieRulesTab(),
    'movie-errors-tab': () => initMovieErrorsTab()
});

// Pure helpers, exposed for tests.
window.vmsamMovie = { movieRegexFor, checkMovieRegex, guessMovieTitle, appendMovieTags, movieSampleFromRegex, movieRenameFor, movieRenameProblem, movieRenamedName };
