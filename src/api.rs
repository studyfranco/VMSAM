use poem::{
    handler,
    http::StatusCode,
    web::{Path, Query},
    Body, IntoResponse, Response, Route,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

// --- Config ---
//
// CREATE_ROOT is the library root as VMSAM itself sees it: folder creation
// prepends it to the relative path chosen in the UI, and VMSAM stores the
// result as the folder's destination_path. Listing a registered folder goes
// the other way (absolute destination_path -> relative under CREATE_ROOT), so
// the two roots must be the same path in both containers.

fn get_create_root() -> PathBuf {
    PathBuf::from(env::var("CREATE_ROOT").unwrap_or_else(|_| "/srv/media".to_string()))
}

fn get_files_root() -> PathBuf {
    PathBuf::from(env::var("FILES_ROOT").unwrap_or_else(|_| "/srv/downloads".to_string()))
}

fn get_vmsam_host() -> String {
    env::var("VMSAM_API_HOST").unwrap_or_else(|_| "http://vmsam-api:8000".to_string())
}

fn json_response(status: StatusCode, body: impl Into<Body>) -> Response {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(body.into())
}

fn detail_response(status: StatusCode, message: String) -> Response {
    let payload = serde_json::json!({ "detail": message });
    json_response(status, payload.to_string())
}

// --- Proxy helpers ---
//
// Every /api/vmsam/* route is a byte-for-byte relay: the upstream status and
// JSON body reach the browser unchanged, so the UI can show VMSAM's own
// `detail` message on a 4xx instead of a generic failure.

async fn relay(result: Result<reqwest::Response, reqwest::Error>) -> Response {
    match result {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body = resp.bytes().await.unwrap_or_default();
            json_response(status, body)
        }
        Err(e) => detail_response(
            StatusCode::BAD_GATEWAY,
            format!("VMSAM API unreachable: {}", e),
        ),
    }
}

async fn proxy_get(upstream_path: &str, params: &HashMap<String, String>) -> Response {
    let client = reqwest::Client::new();
    let mut url = match reqwest::Url::parse(&format!("{}{}", get_vmsam_host(), upstream_path)) {
        Ok(url) => url,
        Err(e) => {
            return detail_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bad VMSAM_API_HOST: {}", e),
            )
        }
    };
    {
        let mut query_pairs = url.query_pairs_mut();
        for (k, v) in params {
            query_pairs.append_pair(k, v);
        }
    }
    relay(client.get(url).send().await).await
}

async fn proxy_post(upstream_path: &str, body: String) -> Response {
    let client = reqwest::Client::new();
    let url = format!("{}{}", get_vmsam_host(), upstream_path);
    relay(
        client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await,
    )
    .await
}

// --- Handlers ---

#[handler]
async fn list_files(Query(params): Query<ListParams>) -> impl IntoResponse {
    let root = if params.root_type.as_deref() == Some("create") {
        get_create_root()
    } else {
        get_files_root()
    };

    let path = params.path.as_deref().unwrap_or("");
    // Prevent directory traversal
    if path.contains("..") {
        return detail_response(StatusCode::BAD_REQUEST, "Invalid path".to_string());
    }

    // A registered folder's destination_path is absolute and starts with the
    // root; strip it so the same call lists a folder VMSAM already knows.
    // Any other leading slash is trimmed so the input is never absolute.
    let root_str = root.to_string_lossy().to_string();
    let relative = path.strip_prefix(root_str.as_str()).unwrap_or(path);
    let safe_path = relative.trim_start_matches('/');
    let full_path = root.join(safe_path);

    if !full_path.starts_with(&root) {
        return detail_response(StatusCode::FORBIDDEN, "Access denied".to_string());
    }

    match tokio::fs::read_dir(full_path).await {
        Ok(mut entries) => {
            let mut items = Vec::new();
            while let Ok(Some(entry)) = entries.next_entry().await {
                let metadata = entry.metadata().await.ok();
                let is_dir = metadata.map(|m| m.is_dir()).unwrap_or(false);
                let name = entry.file_name().to_string_lossy().to_string();

                // Relative path for the frontend: "foo/bar", never "/srv/media/foo/bar"
                let relative_path = entry
                    .path()
                    .strip_prefix(&root)
                    .unwrap_or(&entry.path())
                    .to_string_lossy()
                    .to_string();

                items.push(FileEntry {
                    name,
                    is_dir,
                    path: relative_path,
                });
            }
            // Sort: directories first, then files
            items.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.cmp(&b.name)));

            match serde_json::to_string(&items) {
                Ok(json) => json_response(StatusCode::OK, json),
                Err(e) => detail_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            }
        }
        Err(e) => detail_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[handler]
async fn get_config() -> impl IntoResponse {
    let config = UiConfig {
        create_root: get_create_root().to_string_lossy().to_string(),
        files_root: get_files_root().to_string_lossy().to_string(),
    };
    match serde_json::to_string(&config) {
        Ok(json) => json_response(StatusCode::OK, json),
        Err(e) => detail_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// Rewrite one path field of a JSON body as CREATE_ROOT/<relative path>.
// The browser only ever handles paths relative to the library root; VMSAM
// stores absolute ones. A body that is not JSON is relayed unchanged and
// VMSAM answers 422 itself.
fn absolutise_field(body: &str, field: &str) -> String {
    if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(dest) = json.get(field).and_then(|v| v.as_str()) {
            let relative_dest = dest.trim_start_matches('/');
            let absolute_path = get_create_root().join(relative_dest);
            json[field] = serde_json::Value::String(absolute_path.to_string_lossy().to_string());
            if let Ok(s) = serde_json::to_string(&json) {
                return s;
            }
        }
    }
    body.to_string()
}

fn has_parent_component(body: &str, field: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|json| {
            json.get(field)
                .and_then(|v| v.as_str())
                .map(|p| p.split('/').any(|part| part == ".."))
        })
        .unwrap_or(false)
}

// Proxy for creating folder with absolute path enforcement
#[handler]
async fn proxy_create_folder(body: String) -> impl IntoResponse {
    if has_parent_component(&body, "destination_path") {
        return detail_response(StatusCode::BAD_REQUEST, "Invalid path".to_string());
    }
    proxy_post("/folders/", absolutise_field(&body, "destination_path")).await
}

// Proxy for moving/renaming a folder: same root enforcement on the target
#[handler]
async fn proxy_move_folder(body: String) -> impl IntoResponse {
    if has_parent_component(&body, "new_destination_path") {
        return detail_response(StatusCode::BAD_REQUEST, "Invalid path".to_string());
    }
    proxy_post(
        "/folders/move/",
        absolutise_field(&body, "new_destination_path"),
    )
    .await
}

// Upstream routes with a fixed path: /api/vmsam/<name> -> VMSAM /<path>.
// One table per method, so adding a VMSAM route is one line here.
const GET_ROUTES: &[(&str, &str)] = &[
    ("folders_list", "/folders_list/"),
    ("regex_list", "/regex_folder/"),
    ("special_list", "/special_list/"),
    ("incrementaller_list", "/incrementaller_list/"),
    ("episodes_folder", "/episodes_folder/"),
    ("errors", "/errors"),
    ("health", "/health"),
];

const POST_ROUTES: &[(&str, &str)] = &[
    ("regex", "/regex/"),
    ("special", "/special/"),
    ("incrementaller", "/incrementaller/"),
    ("index_folder", "/index_folder/"),
];

fn lookup(table: &[(&str, &'static str)], name: &str) -> Option<&'static str> {
    table.iter().find(|(n, _)| *n == name).map(|(_, p)| *p)
}

#[handler]
async fn proxy_get_route(
    Path(name): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    match lookup(GET_ROUTES, &name) {
        Some(upstream) => proxy_get(upstream, &params).await,
        None => detail_response(StatusCode::NOT_FOUND, format!("Unknown route: {}", name)),
    }
}

#[handler]
async fn proxy_post_route(Path(name): Path<String>, body: String) -> impl IntoResponse {
    match lookup(POST_ROUTES, &name) {
        Some(upstream) => proxy_post(upstream, body).await,
        None => detail_response(StatusCode::NOT_FOUND, format!("Unknown route: {}", name)),
    }
}

// --- Models ---
#[derive(Deserialize)]
struct ListParams {
    path: Option<String>,
    root_type: Option<String>, // "create" or "files"
}

#[derive(Serialize)]
struct FileEntry {
    name: String,
    is_dir: bool,
    path: String,
}

#[derive(Serialize)]
struct UiConfig {
    create_root: String,
    files_root: String,
}

pub fn routes() -> Route {
    Route::new()
        .at("/fs/list", poem::get(list_files))
        .at("/config", poem::get(get_config))
        .at("/vmsam/folders", poem::post(proxy_create_folder))
        .at("/vmsam/folders_move", poem::post(proxy_move_folder))
        .at(
            "/vmsam/:name",
            poem::get(proxy_get_route).post(proxy_post_route),
        )
}
