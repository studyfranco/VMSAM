// --- Utilities ---

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' :
            type === 'success' ? 'toast-success' :
                'toast-info'
        }`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.classList.add('toast-visible');
    });

    setTimeout(() => {
        toast.classList.remove('toast-visible');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function copyPattern(pattern) {
    navigator.clipboard.writeText(pattern);
    showToast('Pattern copied to clipboard!', 'success');
}

function toggleHelp() {
    const content = document.getElementById('help-content');
    const icon = document.getElementById('help-icon');
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        icon.textContent = '▲';
    } else {
        content.classList.add('hidden');
        icon.textContent = '▼';
    }
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

// State
let state = {
    currentBaseDir: null,
    selectedFolderId: null,
    selectedFolderPath: null,
    regexCards: [], // { id, filename, regex, rename, weight, valid }
    files: []
};

// --- API Client ---

const api = {
    async listFiles(path, rootType = 'files') {
        const p = path ? `?path=${encodeURIComponent(path)}&root_type=${rootType}` : `?root_type=${rootType}`;
        const res = await fetch(`/api/fs/list${p}`);
        if (!res.ok) throw new Error('Failed to list files');
        return res.json();
    },
    async getFolders() {
        const res = await fetch('/api/vmsam/folders_list');
        if (!res.ok) throw new Error('Failed to fetch folders');
        return res.json();
    },
    async createFolder(payload) {
        const res = await fetch('/api/vmsam/folders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Failed to create folder');
        return res.json();
    },
    async createRegex(payload) {
        const res = await fetch('/api/vmsam/regex', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Failed to create regex');
        return res.json();
    }
};

// --- UI Logic: Tabs ---

function switchTab(tabId) {
    document.querySelectorAll('main > section').forEach(el => el.classList.add('hidden'));
    document.getElementById(tabId).classList.remove('hidden');
}

// --- UI Logic: Folder Creation ---

async function loadDirBrowser(path = '') {
    const container = document.getElementById('dir-browser');
    container.innerHTML = '<div class="text-slate-500 text-sm animate-pulse">Loading...</div>';

    try {
        const items = await api.listFiles(path, 'create');
        container.innerHTML = '';

        if (path) {
            const parentPath = path.split('/').slice(0, -1).join('/');
            const backBtn = document.createElement('div');
            backBtn.className = 'dir-item text-blue-300';
            backBtn.innerHTML = '<span>📁 ..</span>';
            backBtn.onclick = () => loadDirBrowser(parentPath);
            container.appendChild(backBtn);
        }

        items.filter(i => i.is_dir).forEach(item => {
            const el = document.createElement('div');
            // Base classes
            el.className = `dir-item group ${state.currentBaseDir === item.path ? 'dir-selected' : ''}`;

            el.innerHTML = `
                <div class="flex-row-center gap-2 flex-1 cursor-pointer" onclick="selectDir('${item.path}', event)">
                    <span>📁 ${item.name}</span>
                </div>
                <div class="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button class="btn btn-sm btn-secondary" onclick="selectDir('${item.path}', event)">Select</button>
                    <button class="btn btn-sm btn-secondary" onclick="loadDirBrowser('${item.path}')">Open →</button>
                </div>
            `;
            container.appendChild(el);
        });
    } catch (e) {
        container.innerHTML = `<div class="text-error text-sm">Error: ${e.message}</div>`;
    }
}

function selectDir(path, event) {
    event.stopPropagation();
    state.currentBaseDir = path;
    updatePreview();

    // Visual feedback
    const container = document.getElementById('dir-browser');
    Array.from(container.children).forEach(child => {
        child.classList.remove('dir-selected');
    });

    // Find the row that contains the button or is the row
    const row = event.target.closest('.dir-item');
    if (row) row.classList.add('dir-selected');

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

    // Construct destination_path logic
    // Format: {baseDir}/tv-shows/{SeriesName} {tvdb-{TVDB_ID}}/{Subfolder}
    const destinationPath = `${state.currentBaseDir}/${seriesName} {tvdb-${tvdbId}} [tvdb-${tvdbId}] [tvdbid-${tvdbId}]/${subfolder}`;

    // Helper safely get int/float
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
        await api.createFolder(payload);
        showToast('Folder created successfully!', 'success');
    } catch (e) {
        showToast('Error creating folder: ' + e.message, 'error');
    }
}

// Listeners
document.getElementById('series-name').addEventListener('input', updatePreview);
document.getElementById('tvdb-id').addEventListener('input', updatePreview);
document.getElementById('subfolder').addEventListener('input', updatePreview);
loadDirBrowser(); // Init

// --- UI Logic: Regex ---

async function initRegexTab() {
    const input = document.getElementById('folder-search');
    const list = document.getElementById('folder-list');

    input.addEventListener('focus', () => list.classList.remove('hidden'));
    // Close when clicking outside - handled by body click? simpler for now

    let folders = [];
    try {
        const response = await api.getFolders();
        folders = response.folders || [];
        if (!folders || folders.length === 0) {
            console.log('No folders returned');
        }
    } catch (e) { console.error(e); }

    const renderFolders = (query) => {
        list.innerHTML = '';
        const filtered = folders.filter(f => f.destination_path.toLowerCase().includes(query.toLowerCase()));
        filtered.forEach(f => {
            const item = document.createElement('div');
            item.className = 'p-3 hover-bg-surface cursor-pointer border-b border-slate-800 text-sm text-muted';
            item.style.borderBottom = '1px solid var(--border)';
            item.textContent = f.destination_path;
            item.onclick = () => {
                state.selectedFolderId = f.id;
                state.selectedFolderPath = f.destination_path;
                document.querySelector('#selected-folder-display span').textContent = f.destination_path;
                document.getElementById('selected-folder-display').classList.remove('hidden');
                document.getElementById('regex-work-area').style.opacity = '1';
                document.getElementById('regex-work-area').style.pointerEvents = 'auto';
                list.classList.add('hidden');
                input.value = '';
            };
            list.appendChild(item);
        });
    };

    input.addEventListener('input', (e) => renderFolders(e.target.value));
    renderFolders('');
}

function clearFolderSelection() {
    state.selectedFolderId = null;
    state.selectedFolderPath = null;
    document.getElementById('selected-folder-display').classList.add('hidden');
    document.getElementById('regex-work-area').style.opacity = '0.5';
    document.getElementById('regex-work-area').style.pointerEvents = 'none';
}

// File Modal
async function openFileModal() {
    const modal = document.getElementById('file-modal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');

    const list = document.getElementById('file-list');
    list.innerHTML = '<div class="text-slate-500 p-4">Loading files...</div>';

    try {
        const files = await api.listFiles('', 'files'); // Root of FILES_ROOT
        // Filter by VIDEO_EXTS (TODO: get from config or hardcode common)
        const vidExts = ['mkv', 'mp4', 'avi', 'm4v', 'mov', 'ts'];
        const vidFiles = files.filter(f => !f.is_dir && vidExts.some(ext => f.name.toLowerCase().endsWith(ext)));

        list.innerHTML = '';
        vidFiles.forEach(f => {
            const row = document.createElement('div');
            row.className = 'flex-row-center gap-3 p-2 hover-bg-surface rounded';
            row.innerHTML = `
                <input type="checkbox" value="${f.name}" class="w-4 h-4">
                <span class="text-sm text-primary">${f.name}</span>
            `;
            list.appendChild(row);
        });
    } catch (e) {
        list.innerHTML = `<div class="text-error p-4">${e.message}</div>`;
    }
}

function closeFileModal() {
    document.getElementById('file-modal').classList.add('hidden');
    document.getElementById('file-modal').classList.remove('flex');
}

function addSelectedFiles() {
    const checks = document.querySelectorAll('#file-list input:checked');
    checks.forEach(c => {
        const filename = c.value;
        addRegexCard(filename);
    });
    closeFileModal();
}

function addRegexCard(filename) {
    const id = Date.now() + Math.random().toString(36).substr(2, 9);
    const container = document.getElementById('regex-cards');

    const card = document.createElement('div');
    card.id = `card-${id}`;
    card.className = 'card p-6 relative group';
    // Style adjustments for inner elements using inline or utility
    card.innerHTML = `
        <button onclick="removeCard('${id}')" class="absolute top-4 right-4 text-muted hover-text-error opacity-0 group-hover:opacity-100 transition-opacity" style="background:none; border:none; cursor:pointer;">🗑️</button>
        <div class="mb-4">
            <div class="text-xs text-muted uppercase font-bold tracking-wider mb-1">Target File</div>
            <div class="font-mono text-sm text-accent surface p-2 break-all">${filename}</div>
        </div>
        
        <div class="grid grid-cols-1 md-grid-cols-2 gap-4">
            <div>
                <label class="label">Regex Pattern</label>
                <input type="text" oninput="validateCard('${id}', '${filename}')" class="regex-input input text-accent font-mono" placeholder="e.g. S\\d{2}E(?P<episode>\\d+)">
                <div class="h-4 mt-1 text-xs text-muted validation-msg"></div>
            </div>
            <div>
                <label class="label">Rename Pattern</label>
                <input type="text" oninput="validateCard('${id}', '${filename}')" class="rename-input input text-blue-300 font-mono" placeholder="e.g. MyShow - {episode_pattern} - Title">
                <div class="h-4 mt-1 text-xs text-muted rename-msg"></div>
            </div>
        </div>
        <div class="mt-4 flex-row-center gap-4">
             <div>
                <label class="label">Weight</label>
                <input type="number" value="1" class="weight-input input text-center" style="width: 5rem;">
            </div>
            <div class="flex-1 surface p-2 flex-between">
                <span class="text-xs text-muted">Episode Extraction:</span>
                <span class="text-sm font-bold text-primary extracted-ep">-</span>
            </div>
        </div>
    `;

    container.appendChild(card);
    state.regexCards.push({ id, filename, el: card });
}

function removeCard(id) {
    const card = document.getElementById(`card-${id}`);
    card.remove();
    state.regexCards = state.regexCards.filter(c => c.id !== id);
}

function validateCard(id, filename) {
    const card = document.getElementById(`card-${id}`);
    const input = card.querySelector('.regex-input').value;
    const msg = card.querySelector('.validation-msg');
    const epDisplay = card.querySelector('.extracted-ep');

    const renameInput = card.querySelector('.rename-input').value;
    const renameMsg = card.querySelector('.rename-msg');

    // Regex Validation
    let regexValid = false;
    try {
        // Rust Regex to JS Regex conversion basics
        // Rust: (?P<name>...) -> JS: (?<name>...)
        let jsRegexStr = input.replace(/\?P<([\w]+)>/g, '?<$1>');
        const re = new RegExp(jsRegexStr);
        const match = filename.match(re);

        if (match && match.groups && match.groups.episode) {
            msg.textContent = "✓ Valid Pattern";
            msg.className = "h-4 mt-1 text-xs text-accent validation-msg";
            epDisplay.textContent = match.groups.episode;
            regexValid = true;
        } else {
            msg.textContent = "⚠ No 'episode' group match";
            msg.className = "h-4 mt-1 text-xs text-error validation-msg"; // Use warning color/error
            epDisplay.textContent = "-";
        }
    } catch (e) {
        msg.textContent = "⚠ Invalid Regex Syntax";
        msg.className = "h-4 mt-1 text-xs text-error validation-msg";
    }

    // Rename Pattern Validation
    if (renameInput && !renameInput.includes('{episode_pattern}')) {
        renameMsg.textContent = "⚠ Must contain {episode_pattern}";
        renameMsg.className = "h-4 mt-1 text-xs text-error rename-msg";
    } else {
        renameMsg.textContent = "";
        renameMsg.className = "h-4 mt-1 text-xs text-muted rename-msg";
    }
}

async function submitAllRegex() {
    const payload = [];

    for (const item of state.regexCards) {
        const card = item.el;
        const regexVal = card.querySelector('.regex-input').value;
        const renameVal = card.querySelector('.rename-input').value;
        const weight = parseInt(card.querySelector('.weight-input').value, 10);

        if (!regexVal) continue;

        payload.push(api.createRegex({
            regex_pattern: regexVal,
            rename_pattern: renameVal,
            weight: weight,
            example_filename: item.filename,
            destination_path: state.selectedFolderPath
        }));
    }

    if (payload.length === 0) return;

    try {
        await Promise.all(payload);
        showToast('All regex rules submitted successfully!', 'success');
        // Clear cards?
        document.getElementById('regex-cards').innerHTML = '';
        state.regexCards = [];
    } catch (e) {
        showToast('Error submitting rules: ' + e.message, 'error');
    }
}

// Init
initRegexTab();
