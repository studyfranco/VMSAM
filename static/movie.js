// VMSAM web interface — Movies section. Vanilla JS, loaded after app.js and
// reusing its helpers (api client helpers, el, showToast, confirmDialog,
// filename engine, folder browser patterns, file modal).
//
// Tabs live in <section id="movie-...-tab"> elements of index.html (between
// the "Movie tabs" comment and MOVIES-SECTIONS-END) and their nav buttons in
// the nav with data-section="movies". Each tab registers its init in
// TAB_INIT (defined in app.js).
//
// A movie is one kept file (a film or a show special):
// `<destination_path>/<title><suffix>.<ext>`. Its rules have no episode group,
// they only have to match the downloaded names. Every call goes through the
// relay /api/movie/<name>; a refusal is shown with VMSAM's own `detail`.
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

// VMSAM answers 404 for an empty list: an empty array here.
async function movieListOr404(name, key, emptyPattern, params) {
    try {
        const response = await movieApi.get(name, params);
        return (response && response[key]) || [];
    } catch (e) {
        if (emptyPattern.test(e.message)) return [];
        throw e;
    }
}

let movieListCache = null;

async function fetchMovies(force = false) {
    if (movieListCache && !force) return movieListCache;
    movieListCache = await movieListOr404('movies_list', 'movies', /No movies found/i);
    return movieListCache;
}

function fetchMovieFolders() {
    return movieListOr404('folders_list', 'folders', /No folders found/i);
}

function fetchMovieRules(movieId) {
    return movieListOr404('regex_movie', 'regex_patterns', /No regex found/i, { movie_id: movieId });
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

function movieFileName(movie, extension) {
    return `${movie.title}${movie.suffix || ''}${extension || ''}`;
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

function movieLabel(movie) {
    return `${movie.title}${movie.suffix || ''}`;
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
// onSelect(item), onClear(). picker.select(item, true) selects silently.
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
        const items = picker.items.filter(item => !query || options.text(item).toLowerCase().includes(query));
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
    row.appendChild(el('span', 'text-muted text-xs font-mono', `${movieRel(movie.destination_path)} · #${movie.id}`));
    return row;
}

function createMoviePicker(host, onSelect, onClear) {
    return createMovieSearchPicker(host, {
        placeholder: 'Search a movie (title, suffix, directory, id)…',
        emptyText: 'No movie yet: create one first',
        load: () => fetchMovies(),
        text: m => `${movieLabel(m)} ${m.destination_path} #${m.id}`,
        render: moviePickerRow,
        chosen: m => `${movieLabel(m)} — ${movieRel(m.destination_path)} (#${m.id})`,
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

// A directory browser over CREATE_ROOT with a name filter (applyDirFilter).
// options: containerId, filterId, hereText / hereTitle (button of the listed
// directory: onPick(currentDir)), onPick(path) (click on a row), markRows()
// after each listing. "Open →" or a double click navigates.
function createMovieDirBrowser(options) {
    const browser = { dir: '' };
    const container = document.getElementById(options.containerId);
    document.getElementById(options.filterId).addEventListener('input', () => applyDirFilter(options.containerId, options.filterId));

    browser.load = async (path = '', fallbackToRoot = false) => {
        container.innerHTML = '';
        container.appendChild(el('div', 'text-muted text-center p-4', 'Loading...'));
        try {
            const items = await api.listFiles(path, 'create');
            browser.dir = path;
            container.innerHTML = '';
            const where = el('div', 'dir-item register-where flex-between');
            where.appendChild(el('span', 'font-mono text-muted', path ? `/${path}` : '/ (library root)'));
            where.appendChild(movieButton(options.hereText, 'btn btn-sm btn-secondary movie-browser-here', () => options.onPick(path), options.hereTitle));
            container.appendChild(where);
            if (path) {
                const back = el('div', 'dir-item text-blue');
                back.appendChild(el('span', null, '📁 ..'));
                back.onclick = () => browser.load(path.split('/').slice(0, -1).join('/'));
                container.appendChild(back);
            }
            const dirs = items.filter(i => i.is_dir);
            if (!dirs.length) container.appendChild(el('div', 'text-muted text-center p-2', 'No sub-folder here'));
            dirs.forEach(item => {
                const row = el('div', 'dir-item');
                row.dataset.path = item.path;
                row.dataset.name = item.name;
                row.onclick = () => options.onPick(item.path);
                row.ondblclick = () => browser.load(item.path);
                const name = el('div', 'dir-name flex-1');
                name.appendChild(el('span', null, `📁 ${item.name}`));
                const actions = el('div', 'dir-actions');
                actions.appendChild(movieButton('Open →', 'btn btn-sm btn-secondary', (e) => { e.stopPropagation(); browser.load(item.path); }));
                row.append(name, actions);
                container.appendChild(row);
            });
            if (dirs.length) applyDirFilter(options.containerId, options.filterId);
            if (options.markRows) options.markRows();
        } catch (e) {
            if (path && fallbackToRoot) { browser.load(''); return; }
            container.innerHTML = '';
            const err = el('div', 'text-danger text-sm p-4 text-center', `Error: ${e.message}`);
            if (path) {
                err.appendChild(document.createElement('br'));
                err.appendChild(movieButton('Back to the library root', 'btn btn-sm btn-secondary mt-2', () => browser.load('')));
            }
            container.appendChild(err);
        }
    };
    return browser;
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

function markMovieCreateRows() {
    const dest = cleanRelativePath(document.getElementById('movie-create-dest').value);
    document.querySelectorAll('#movie-create-browser .dir-item[data-path]').forEach(row => row.classList.toggle('selected', row.dataset.path === dest));
}

function initMovieCreateTab() {
    if (movieCreateTab) { movieCreateTab.browser.load(movieCreateTab.browser.dir, true); updateMovieCreatePreview(); return; }
    const browser = createMovieDirBrowser({
        containerId: 'movie-create-browser',
        filterId: 'movie-create-filter',
        hereText: 'Use this directory',
        hereTitle: 'The movie goes directly in the directory listed',
        onPick: (path) => setMovieCreateField('movie-create-dest', path),
        markRows: markMovieCreateRows
    });
    const meta = createMovieMetadataGenerator(document.getElementById('movie-create-meta'), [
        { text: 'Use as suffix', title: 'Suffix = a space and the tags', use: tags => setMovieCreateField('movie-create-suffix', ' ' + tags) },
        { text: 'Append to suffix', title: 'Adds the tags at the end of the suffix', use: tags => setMovieCreateField('movie-create-suffix', appendMovieTags(document.getElementById('movie-create-suffix').value, tags)) },
        {
            text: 'Append to the destination folder name',
            title: 'Adds the tags at the end of the last directory of the destination',
            use: tags => {
                const dest = cleanRelativePath(document.getElementById('movie-create-dest').value);
                if (!dest) { showToast('Choose or type the destination directory first', 'error'); return; }
                setMovieCreateField('movie-create-dest', appendMovieTags(dest, tags).trim());
            }
        }
    ], updateMovieCreatePreview);
    movieCreateTab = { browser, meta, returnGroupId: null };
    ['movie-create-dest', 'movie-create-title', 'movie-create-suffix'].forEach(id => document.getElementById(id).addEventListener('input', () => {
        updateMovieCreatePreview();
        if (id === 'movie-create-dest') markMovieCreateRows();
    }));
    document.getElementById('movie-create-btn').onclick = submitMovieCreate;
    document.getElementById('movie-create-back').onclick = () => switchTab('movie-regex-tab');
    fetchUiConfig().then(config => { document.getElementById('movie-create-root').textContent = config.create_root; })
        .catch(e => showToast('Cannot read the library root: ' + e.message, 'error'));
    browser.load('');
    updateMovieCreatePreview();
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

function updateMovieCreatePreview() {
    const preview = document.getElementById('movie-create-preview');
    const { dest, title, suffix } = movieCreateValues();
    const problems = [];
    if (!title.trim()) problems.push('the title is empty');
    if (/\//.test(title) || /\//.test(suffix)) problems.push('the title and the suffix cannot hold a /');
    if (title.startsWith('.')) problems.push('a title starting with a dot makes a hidden file');
    if (dest.split('/').includes('..')) problems.push('".." is refused in the directory');
    preview.innerHTML = '';
    preview.appendChild(el('span', 'text-muted', (dest || '(library root)') + '/'));
    preview.appendChild(el('span', 'text-main', `${title || '<title>'}${suffix}`));
    preview.appendChild(el('span', 'text-muted', '.<ext>'));
    if (problems.length) preview.appendChild(el('div', 'text-warning text-xs mt-1', '⚠ ' + problems.join('; ')));
    document.getElementById('movie-create-btn').disabled = problems.length > 0;
}

// Opens the creation tab with a title (from a Bulk Regex card); the movie
// created next is linked to that card.
function startMovieCreation(title, groupId) {
    switchTab('movie-create-tab');
    setMovieCreateField('movie-create-title', title || '');
    movieCreateTab.returnGroupId = groupId === undefined ? null : groupId;
    document.getElementById('movie-create-back').classList.toggle('hidden', movieCreateTab.returnGroupId === null);
    document.getElementById('movie-create-title').focus();
}

async function submitMovieCreate() {
    const { dest, title, suffix } = movieCreateValues();
    const number = (id, def, parse) => { const v = parse(document.getElementById(id).value); return isNaN(v) ? def : v; };
    const payload = {
        destination_path: dest,
        title,
        suffix,
        original_language: document.getElementById('movie-create-lang').value.trim() || 'en',
        number_cut: number('movie-create-cut', 10, v => parseInt(v, 10)),
        cut_file_to_get_delay_second_method: number('movie-create-ratio', 2.0, parseFloat)
    };
    const ids = movieCreateTab.meta.getIds();
    if (Object.keys(ids).length) payload.metadata_ids = ids;
    const button = document.getElementById('movie-create-btn');
    button.disabled = true;
    try {
        const result = await movieApi.post('movies', payload);
        showToast(`${result.message || 'Movie created'} (#${result.movie_id})`, 'success', 5000);
        invalidateMovies();
        const movie = (await fetchMovies(true)).find(m => m.id === result.movie_id);
        if (movieCreateTab.returnGroupId !== null && movie) {
            linkMovieToRegexGroup(movieCreateTab.returnGroupId, movie);
            showToast('Linked to its Bulk Regex group: "← Back to Bulk Regex"', 'info', 5000);
        }
        movieCreateTab.browser.load(movieCreateTab.browser.dir, true);
    } catch (e) {
        showToast('Movie not created: ' + e.message, 'error', 8000);
    } finally {
        updateMovieCreatePreview();
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

function newMovieRegexGroup(files) {
    return { id: movieRegexTab.nextId++, files: [...files], regexManual: null, movie: null, weight: 1, saved: false, error: null, result: null };
}

function movieRegexOf(group) {
    return group.regexManual !== null ? group.regexManual : movieRegexFor(group.files);
}

async function addMovieRegexFiles(names) {
    if (!movieRegexTab) initMovieRegexTab();
    const known = new Set([...movieRegexTab.groups.flatMap(g => g.files), ...movieRegexTab.covered.map(c => c.name)]);
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
        row.append(left, el('div', 'badge', `W: ${rule.weight}`));
        content.appendChild(row);
    });
}

function moveMovieRegexFile(name, fromId, toId) {
    if (fromId === toId) return;
    const from = movieRegexTab.groups.find(g => g.id === fromId);
    if (!from) return;
    from.files = from.files.filter(f => f !== name);
    from.saved = false;
    from.regexManual = null;
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
    const groups = movieRegexTab.groups;
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
        movieButton('Auto-group', 'btn btn-secondary btn-sm board-auto-group', () => { movieRegexTab.groups = groupByNamePattern(allFiles()).map(newMovieRegexGroup); renderMovieRegexBoard(); }, 'Regroup every file by the words its name shares with the others (manual edits are dropped)'),
        movieButton('Split all', 'btn btn-secondary btn-sm board-split-all', () => { movieRegexTab.groups = allFiles().map(n => newMovieRegexGroup([n])); renderMovieRegexBoard(); }, 'One file per card (manual edits are dropped)'),
        movieButton('Clear', 'btn btn-secondary btn-sm', () => { movieRegexTab.groups = []; movieRegexTab.covered = []; renderMovieRegexCovered(); renderMovieRegexBoard(); }, 'Remove every file from the board')
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
    const save = movieButton('Save Rule', 'btn btn-primary btn-sm movie-regex-save', () => saveMovieRegexGroup(group, regexInput.value, save));
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
        save.disabled = !check.allMatch || !group.movie;
    }

    function refreshMovie() {
        finalLine.innerHTML = '';
        existing.innerHTML = '';
        refreshCheck();
        if (!group.movie) {
            finalLine.appendChild(el('span', 'text-muted', 'Link the group to a movie.'));
            return;
        }
        const movie = group.movie;
        finalLine.append(el('span', 'text-muted', 'Becomes '), el('span', 'font-mono text-success', `${movieRel(movie.destination_path)}/${movieFileName(movie, movieFileExtension(group.files[0]))}`));
        if (movie.file_path) finalLine.appendChild(el('span', 'text-muted', ` (the movie already has ${movieBaseName(movie.file_path)}: a newcomer is merged with it)`));
        loadMovieRulesFor(movie.id).then(rules => {
            if (group.movie !== movie) return;
            existing.innerHTML = '';
            existing.appendChild(el('div', 'text-xs text-muted uppercase mt-2', rules.length ? `Existing rules of this movie (${rules.length})` : 'This movie has no rule yet'));
            const selection = [...movieRegexTab.groups.flatMap(g => g.files), ...movieRegexTab.covered.map(c => c.name)];
            rules.forEach(rule => {
                const row = el('div', 'rule-row');
                const left = el('div', 'rule-row-patterns');
                left.appendChild(el('div', 'font-mono text-accent', rule.regex_pattern));
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
                row.append(left, el('div', 'badge', `W: ${rule.weight}`));
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

async function saveMovieRegexGroup(group, pattern, button) {
    if (!group.movie) { showToast('Link the group to a movie first', 'error'); return; }
    if (!checkMovieRegex(pattern, group.files).allMatch) { showToast('The regex must match every file of the group', 'error'); return; }
    button.disabled = true;
    try {
        const result = await movieApi.post('regex', { regex_pattern: pattern, movie_id: group.movie.id, weight: group.weight, example_filename: group.files[0] });
        group.saved = true;
        group.error = null;
        group.regexManual = pattern;
        group.result = `${result.message || 'Regex saved'} → ${result.final_file_name}`;
        showToast(group.result, 'success', 5000);
        movieRegexTab.rulesByMovie.delete(group.movie.id);
    } catch (e) {
        group.saved = false;
        group.error = e.message;
        showToast('Rule not saved: ' + e.message, 'error', 8000);
    } finally {
        renderMovieRegexBoard();
    }
}

// --- M4c. Index tab ---

let movieIndexTab = null;

function initMovieIndexTab() {
    if (movieIndexTab) { if (movieIndexTab.movie) showIndexMovie(movieIndexTab.movie.id); return; }
    movieIndexTab = { movie: null, picker: null, dirPicker: null };
    movieIndexTab.picker = createMoviePicker(document.getElementById('movie-index-picker'), (movie) => showIndexMovie(movie.id), clearIndexMovie);
    movieIndexTab.dirPicker = createMovieFolderPicker(document.getElementById('movie-index-dir-picker'), showIndexDirMovies, () => { document.getElementById('movie-index-dir-movies').innerHTML = ''; });
    document.getElementById('movie-index-reload').onclick = () => { if (movieIndexTab.movie) showIndexMovie(movieIndexTab.movie.id); };
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
    if (!movieIndexTab.picker.selected || movieIndexTab.picker.selected.id === movie.id) movieIndexTab.picker.select(movie, true);

    info.innerHTML = '';
    [
        ['Directory', movieRel(movie.destination_path)],
        ['Final name', movieFileName(movie, '.<ext>')],
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
        row.append(el('div', 'font-mono text-accent rule-row-patterns', rule.regex_pattern), el('div', 'badge', `W: ${rule.weight}`));
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
        if (owner && owner.id === movie.id) statusCell.appendChild(el('span', 'badge badge-success', 'registered file of this movie'));
        else if (owner) statusCell.appendChild(el('span', 'badge', `file of movie #${owner.id} ${movieLabel(owner)}`));
        else if (rule) {
            statusCell.appendChild(el('span', 'badge badge-success', `matched (W ${rule.weight})`));
            if (!isVideoName(file.name)) statusCell.appendChild(el('span', 'badge badge-danger', 'not a video'));
            statusCell.appendChild(el('div', 'font-mono text-xs text-muted', rule.regex_pattern));
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
                if (!formCell.children.length) formCell.appendChild(renderInlineMovieRule(movie, file.name, () => showIndexMovie(movie.id)));
            }, 'Add a rule of this movie matching this file'));
            table.appendChild(formRow);
        }
    });
    filesBox.appendChild(table);
}

// Inline form: a rule of `movie` built from one file name.
function renderInlineMovieRule(movie, fileName, onSaved) {
    const box = el('div', 'movie-inline-rule');
    const input = movieInput('text', 'input font-mono pattern-input movie-inline-pattern', movieRegexFor([fileName]));
    const weight = movieInput('number', 'input text-center small-number', 1);
    weight.min = 1;
    const message = el('div', 'text-xs mt-1');
    const error = el('div', 'movie-message hidden');
    const save = movieButton('Save rule', 'btn btn-primary btn-sm movie-inline-save', async () => {
        save.disabled = true;
        try {
            const result = await movieApi.post('regex', { regex_pattern: input.value, movie_id: movie.id, weight: parseInt(weight.value, 10) || 1, example_filename: fileName });
            showToast(`${result.message || 'Regex saved'} → ${result.final_file_name}`, 'success', 5000);
            onSaved();
        } catch (e) {
            movieMessage(error, e.message, 'danger');
            showToast('Rule not saved: ' + e.message, 'error', 8000);
            save.disabled = false;
        }
    });
    const check = () => {
        const ok = checkMovieRegex(input.value, [fileName]).allMatch;
        message.textContent = ok ? '✓ Matches the file' : '⚠ Does not match the file';
        message.className = `text-xs mt-1 ${ok ? 'text-success' : 'text-warning'}`;
        save.disabled = !ok;
    };
    input.oninput = check;
    const row = el('div', 'flex items-center movie-inline-row');
    row.append(input, weight, save);
    box.append(el('label', 'label', `New rule of ${movieLabel(movie)} (pattern, weight)`), row, message, error);
    check();
    return box;
}

const MOVIE_SKIP_REASONS = {
    not_a_video_file: 'not a video file',
    movie_already_has_a_file: 'the movie already has a file (nothing is merged here)',
    not_selected: 'another file was taken (heavier rule or larger file)',
    target_exists: 'the final name already exists on disk',
    target_registered: 'the final name is another movie\'s file',
    rename_failed: 'the rename failed'
};

async function runIndexMovie() {
    const movie = movieIndexTab.movie;
    if (!movie) return;
    const button = document.getElementById('movie-index-run');
    const report = document.getElementById('movie-index-report');
    button.disabled = true;
    try {
        const result = await movieApi.post('index_movie', { movie_id: movie.id });
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
        await showIndexMovie(movie.id);
    } catch (e) {
        showToast('Indexing refused: ' + e.message, 'error', 8000);
    } finally {
        button.disabled = false;
    }
}

// --- M4d. Move Folder tab ---

let movieMoveTab = null;

function initMovieMoveTab() {
    if (movieMoveTab) { movieMoveTab.browser.load(movieMoveTab.browser.dir, true); return; }
    const picker = createMovieFolderPicker(document.getElementById('movie-move-picker'), (folder) => { movieMoveTab.shown = showMovieMoveFolder(folder); }, () => {
        movieMoveTab.folder = null;
        movieMoveTab.relative = null;
        document.getElementById('movie-move-current').classList.add('hidden');
        document.getElementById('movie-move-work').classList.add('disabled-area');
        document.getElementById('movie-move-path').value = '';
        updateMovieMovePreview();
    });
    const browser = createMovieDirBrowser({
        containerId: 'movie-move-browser',
        filterId: 'movie-move-filter',
        hereText: 'Move here',
        hereTitle: 'Move the directory directly inside this one',
        onPick: setMovieMoveInside,
        markRows: markMovieMoveRows
    });
    movieMoveTab = { picker, browser, folder: null, relative: null, keepReport: false, shown: null };
    document.getElementById('movie-move-path').addEventListener('input', updateMovieMovePreview);
    document.getElementById('movie-move-btn').onclick = moveMovieFolder;
    fetchUiConfig().then(config => { document.getElementById('movie-move-root').textContent = config.create_root; })
        .catch(e => showToast('Cannot read the library root: ' + e.message, 'error'));
    browser.load('');
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
    markMovieMoveRows();
    const moviesBox = document.getElementById('movie-move-movies');
    moviesBox.innerHTML = '';
    try {
        const own = normalizePath(folder.destination_path);
        const movies = (await fetchMovies(true)).filter(m => normalizePath(m.destination_path) === own || normalizePath(m.destination_path).startsWith(own + '/'));
        if (movieMoveTab.folder !== folder) return;
        moviesBox.appendChild(el('div', 'text-muted', `Movies moved with it (${movies.length}):`));
        const list = el('ul', 'movie-plain-list');
        movies.forEach(m => list.appendChild(el('li', 'font-mono', `#${m.id} ${movieLabel(m)}${normalizePath(m.destination_path) !== own ? ' — in ' + movieRel(m.destination_path) : ''}`)));
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
    document.querySelectorAll('#movie-move-browser .dir-item[data-path]').forEach(row => row.classList.toggle('selected', row.dataset.path === dir));
    updateMovieMovePreview();
}

function markMovieMoveRows() {
    if (!movieMoveTab) return;
    document.querySelectorAll('#movie-move-browser .dir-item[data-path]').forEach(row => {
        const isCurrent = movieMoveTab.relative !== null && row.dataset.path === movieMoveTab.relative;
        row.classList.toggle('dir-registered', isCurrent);
        const badge = row.querySelector('.register-badge');
        if (isCurrent && !badge) row.querySelector('.dir-name').appendChild(el('span', 'badge badge-success register-badge', 'this directory'));
        if (!isCurrent && badge) badge.remove();
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
        const moved = (result.movies_updated || []).map(id => movies.find(m => m.id === id)).filter(Boolean);
        showToast(`${result.message || 'Folder moved'}: ${moved.length} movie${moved.length === 1 ? '' : 's'}, ${result.files_updated} file path${result.files_updated === 1 ? '' : 's'} updated`, 'success', 6000);
        report.innerHTML = '';
        report.appendChild(el('div', 'text-muted', 'Moved movies:'));
        const list = el('ul', 'font-mono move-removed-list movie-move-moved');
        moved.forEach(m => list.appendChild(el('li', null, `#${m.id} ${movieLabel(m)} → ${movieRel(m.destination_path)}`)));
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
        tab.browser.load(tab.browser.dir, true);
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
    return [movie.id, movie.title, movie.suffix, movie.destination_path, movie.file_path || '', ...Object.entries(movie.metadata_ids || {}).map(([k, v]) => `${k} ${v}`)].join(' ').toLowerCase();
}

function renderMetadataChips(ids) {
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
        row.dataset.movieId = String(movie.id);
        const idsCell = el('td');
        idsCell.appendChild(renderMetadataChips(movie.metadata_ids));
        const actions = el('td', 'movie-actions-cell');
        actions.append(
            movieButton('Edit', 'btn btn-secondary btn-sm movie-edit-btn', () => openMovieEdit(movie)),
            movieButton('Delete', 'btn btn-secondary btn-sm movie-delete-btn', () => deleteMovie(movie))
        );
        row.append(
            el('td', 'text-muted', String(movie.id)),
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

function openMovieEdit(movie) {
    const card = document.getElementById('movie-edit');
    card.innerHTML = '';
    card.classList.remove('hidden');
    card.dataset.movieId = String(movie.id);
    const header = el('div', 'flex flex-between items-center');
    header.append(el('h2', 'section-title', `Edit movie #${movie.id}`), movieButton('✕', 'btn-icon', () => card.classList.add('hidden'), 'Close'));
    card.appendChild(header);
    card.appendChild(el('div', 'text-sm text-muted mb-2', `Directory: ${movieRel(movie.destination_path)} (moving it is the Move Folder tab)`));
    const title = movieInput('text', 'input movie-edit-title', movie.title);
    const suffix = movieInput('text', 'input font-mono movie-edit-suffix', movie.suffix || '');
    const lang = movieInput('text', 'input movie-edit-lang', movie.original_language);
    const cut = movieInput('number', 'input movie-edit-cut', movie.number_cut);
    const ratio = movieInput('number', 'input movie-edit-ratio', movie.cut_file_to_get_delay_second_method);
    ratio.step = '0.1';
    const grid = el('div', 'grid-2');
    grid.append(movieLabelled('Title', title), movieLabelled('Suffix', suffix));
    const grid3 = el('div', 'grid-3');
    grid3.append(movieLabelled('Original Language', lang), movieLabelled('Number Cut', cut), movieLabelled('Cut Ratio (1/N)', ratio));
    const metaHost = el('div', 'movie-edit-meta');
    const preview = el('div', 'text-sm mt-2 font-mono movie-edit-preview');
    const error = el('div', 'movie-message hidden');
    card.append(grid, grid3, el('div', 'label mt-2', 'Metadata ids'), metaHost, preview, error);
    const updatePreview = () => {
        preview.innerHTML = '';
        const next = `${title.value}${suffix.value}`;
        const changed = next !== movieLabel(movie);
        if (movie.file_path) {
            preview.append(el('span', 'text-muted', 'File: '), el('span', null, movieBaseName(movie.file_path)));
            if (changed) preview.append(el('span', 'text-muted', ' → '), el('span', 'text-success', next + movieFileExtension(movie.file_path)), el('span', 'text-muted', ' (renamed on disk)'));
        } else {
            preview.append(el('span', 'text-muted', 'Final name: '), el('span', changed ? 'text-success' : null, next + '.<ext>'), el('span', 'text-muted', ' (no file registered yet)'));
        }
    };
    const meta = createMovieMetadataGenerator(metaHost, [
        { text: 'Use as suffix', use: tags => { suffix.value = ' ' + tags; updatePreview(); } },
        { text: 'Append to suffix', use: tags => { suffix.value = appendMovieTags(suffix.value, tags); updatePreview(); } }
    ]);
    meta.setIds(movie.metadata_ids || {});
    [title, suffix].forEach(input => input.addEventListener('input', updatePreview));
    updatePreview();
    const save = movieButton('Save changes', 'btn btn-primary movie-edit-save', async () => {
        const payload = { movie_id: movie.id };
        if (title.value !== movie.title) payload.title = title.value;
        if (suffix.value !== (movie.suffix || '')) payload.suffix = suffix.value;
        if (lang.value.trim() && lang.value.trim() !== movie.original_language) payload.original_language = lang.value.trim();
        if (cut.value !== '' && parseInt(cut.value, 10) !== movie.number_cut) payload.number_cut = parseInt(cut.value, 10);
        if (ratio.value !== '' && parseFloat(ratio.value) !== movie.cut_file_to_get_delay_second_method) payload.cut_file_to_get_delay_second_method = parseFloat(ratio.value);
        const ids = meta.getIds();
        const sortedJson = obj => JSON.stringify(Object.keys(obj).sort().map(k => [k, obj[k]]));
        if (sortedJson(ids) !== sortedJson(movie.metadata_ids || {})) payload.metadata_ids = ids;
        if (Object.keys(payload).length === 1) { showToast('Nothing changed', 'info'); return; }
        save.disabled = true;
        movieMessage(error, '');
        try {
            const result = await movieApi.post('movies_update', payload);
            const renamed = result.file_renamed ? ` — file renamed to ${movieBaseName(result.file_path)}` : '';
            showToast(`${result.message || 'Movie updated'}${renamed}`, 'success', 6000);
            invalidateMovies();
            await loadMovieList();
            openMovieEdit(result);
        } catch (e) {
            movieMessage(error, e.message, 'danger');
            showToast('Not saved: ' + e.message, 'error', 8000);
            save.disabled = false;
        }
    });
    const footer = el('div', 'movie-actions mt-2');
    footer.append(movieButton('Cancel', 'btn btn-secondary', () => card.classList.add('hidden')), save);
    card.appendChild(footer);
    card.scrollIntoView({ block: 'start' });
}

async function deleteMovie(movie) {
    const confirmed = await confirmDialog(`#${movie.id} ${movieLabel(movie)}\n${movieRel(movie.destination_path)}${movie.file_path ? '\nIts file stays on disk: ' + movieBaseName(movie.file_path) : ''}`, { title: 'Delete this movie?', okText: 'Delete' });
    if (!confirmed) return;
    try {
        const result = await movieApi.post('movies_delete', { movie_id: movie.id });
        showToast(result.message || 'Movie deleted', 'success', 6000);
        invalidateMovies();
        const card = document.getElementById('movie-edit');
        if (card.dataset.movieId === String(movie.id)) card.classList.add('hidden');
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
    const rules = all.filter(r => !query || [r.regex_pattern, r.title, r.suffix, r.destination_path, '#' + r.movie_id].join(' ').toLowerCase().includes(query));
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
    ['Pattern', 'Movie', 'Directory', 'W', ''].forEach(h => head.appendChild(el('th', null, h)));
    table.appendChild(head);
    rules.forEach(rule => {
        const row = el('tr', testName && movieRegexMatches(rule.regex_pattern, testName) ? 'movie-row-hit' : null);
        row.dataset.pattern = rule.regex_pattern;
        const actions = el('td', 'movie-actions-cell');
        actions.append(
            movieButton('Edit', 'btn btn-secondary btn-sm movie-rule-edit', () => { movieRulesTab.editing = movieRulesTab.editing === rule.regex_pattern ? null : rule.regex_pattern; renderMovieRules(); }),
            movieButton('Delete', 'btn btn-secondary btn-sm movie-rule-delete', () => deleteMovieRule(rule))
        );
        row.append(
            el('td', 'font-mono text-accent', rule.regex_pattern),
            el('td', null, `#${rule.movie_id} ${movieLabel(rule)}`),
            el('td', 'font-mono text-muted', movieRel(rule.destination_path)),
            el('td', null, String(rule.weight)),
            actions
        );
        table.appendChild(row);
        if (movieRulesTab.editing === rule.regex_pattern) {
            const formRow = el('tr', 'movie-inline-form-row');
            const cell = el('td');
            cell.colSpan = 5;
            cell.appendChild(renderMovieRuleEditor(rule, testName));
            formRow.appendChild(cell);
            table.appendChild(formRow);
        }
    });
    box.appendChild(table);
}

function renderMovieRuleEditor(rule, testName) {
    const box = el('div', 'movie-inline-rule movie-rule-editor');
    const pattern = movieInput('text', 'input font-mono pattern-input movie-rule-pattern', rule.regex_pattern);
    const weight = movieInput('number', 'input text-center small-number movie-rule-weight', rule.weight);
    weight.min = 1;
    const example = movieInput('text', 'input font-mono movie-rule-example', testName && movieRegexMatches(rule.regex_pattern, testName) ? testName : '');
    example.placeholder = 'Example file name the rule must match (required by VMSAM)';
    const check = el('div', 'text-xs mt-1');
    const error = el('div', 'movie-message hidden');
    let target = { id: rule.movie_id, title: rule.title, suffix: rule.suffix, destination_path: rule.destination_path };
    const pickerHost = el('div');
    const picker = createMoviePicker(pickerHost, (movie) => { target = movie; }, () => { target = null; });
    picker.select(target, true);
    const save = movieButton('Save', 'btn btn-primary btn-sm movie-rule-save', async () => {
        if (!target) { showToast('Choose the movie', 'error'); return; }
        const payload = { regex_pattern: rule.regex_pattern, example_filename: example.value };
        if (pattern.value !== rule.regex_pattern) payload.new_regex_pattern = pattern.value;
        if (target.id !== rule.movie_id) payload.movie_id = target.id;
        const w = parseInt(weight.value, 10);
        if (!isNaN(w) && w !== rule.weight) payload.weight = w;
        save.disabled = true;
        try {
            const result = await movieApi.post('regex_update', payload);
            showToast(`${result.message || 'Regex updated'} → ${result.final_file_name}`, 'success', 5000);
            movieRulesTab.editing = null;
            if (movieRegexTab) movieRegexTab.rulesByMovie.clear();
            loadMovieRules();
        } catch (e) {
            movieMessage(error, e.message, 'danger');
            showToast('Not saved: ' + e.message, 'error', 8000);
            save.disabled = false;
        }
    });
    const refresh = () => {
        const result = checkMovieRegex(pattern.value, example.value ? [example.value] : []);
        if (result.syntaxError) { check.textContent = '⚠ Invalid regex: ' + result.syntaxError; check.className = 'text-xs mt-1 text-danger'; }
        else if (!example.value) { check.textContent = 'Type an example file name: VMSAM checks the rule against it'; check.className = 'text-xs mt-1 text-muted'; }
        else if (result.allMatch) { check.textContent = '✓ Matches the example'; check.className = 'text-xs mt-1 text-success'; }
        else { check.textContent = '⚠ Does not match the example'; check.className = 'text-xs mt-1 text-warning'; }
        save.disabled = !(result.allMatch && example.value);
    };
    pattern.oninput = refresh;
    example.oninput = refresh;
    const fields = el('div', 'movie-rule-fields');
    fields.append(movieLabelled('Pattern', pattern), movieLabelled('Weight', weight));
    const footer = el('div', 'movie-actions');
    footer.append(movieButton('Cancel', 'btn btn-secondary btn-sm', () => { movieRulesTab.editing = null; renderMovieRules(); }), save);
    box.append(fields, movieLabelled('Movie', pickerHost), movieLabelled('Example file name', example), check, error, footer);
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
        movieErrorsTab = { timer: null };
        document.getElementById('movie-errors-reload').onclick = () => loadMovieErrors();
    }
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
    let errors, movies;
    try {
        await fetchUiConfig();
        const response = await movieApi.get('errors');
        errors = (response && response.incompatible_files) || [];
        movies = await fetchMovies(true);
    } catch (e) {
        box.innerHTML = '';
        box.appendChild(el('div', 'text-danger text-sm', e.message));
        return;
    }
    document.getElementById('movie-errors-count').textContent = String(errors.length);
    box.innerHTML = '';
    if (!errors.length) { box.appendChild(el('div', 'text-muted text-sm', 'No incompatible file')); return; }
    const byMovie = new Map();
    errors.forEach(item => {
        if (!byMovie.has(item.movie_id)) byMovie.set(item.movie_id, []);
        byMovie.get(item.movie_id).push(item);
    });
    byMovie.forEach((items, movieId) => {
        const first = items[0];
        const movie = movies.find(m => m.id === movieId);
        const group = el('div', 'card movie-error-group');
        group.dataset.movieId = String(movieId);
        const header = el('div', 'flex flex-between items-center');
        header.append(el('h3', 'movie-error-title', `#${movieId} ${movieLabel(first)}`), el('span', 'badge', `${items.length} file${items.length > 1 ? 's' : ''}`));
        group.append(
            header,
            el('div', 'text-sm text-muted font-mono', movieRel(first.destination_path)),
            el('div', 'text-sm mb-2', movie && movie.file_path ? `Registered file: ${movieBaseName(movie.file_path)} (weight ${movie.file_weight})` : 'No registered file: a fusion has nothing to merge with')
        );
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
        group.appendChild(table);
        box.appendChild(group);
    });
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
    const confirmed = await confirmDialog(`${item.file_path}\nThe file, its logs and the row are deleted, then its directory if no video is left in it.`, { title: 'Delete this incompatible file?', okText: 'Delete' });
    if (!confirmed) return;
    try {
        const result = await movieApi.post('errors_delete', { error_file_path: item.file_path });
        const parts = [`${(result.deleted_files || []).length} file(s) deleted`];
        if (result.file_was_missing) parts.push('the file was already missing');
        if ((result.removed_directories || []).length) parts.push(`directories removed: ${result.removed_directories.join(', ')}`);
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
window.vmsamMovie = { movieRegexFor, checkMovieRegex, guessMovieTitle, appendMovieTags };
