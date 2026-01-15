use poem::{
    handler,
    web::Query,
    IntoResponse, Route, Body, Response, http::StatusCode
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::env;


// --- Config ---

fn get_create_root() -> PathBuf {
    PathBuf::from(env::var("CREATE_ROOT").unwrap_or_else(|_| "/srv/media".to_string()))
}

fn get_files_root() -> PathBuf {
    PathBuf::from(env::var("FILES_ROOT").unwrap_or_else(|_| "/srv/downloads".to_string()))
}

fn get_vmsam_host() -> String {
    env::var("VMSAM_API_HOST").unwrap_or_else(|_| "http://vmsam-api:8000".to_string())
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
         return Response::builder().status(StatusCode::BAD_REQUEST).body(Body::from("Invalid path"));
    }
    
    // Ensure we don't treat the input as absolute path by trimming leading slashes
    let safe_path = path.trim_start_matches('/');
    let full_path = root.join(safe_path);

    if !full_path.starts_with(&root) {
        return Response::builder().status(StatusCode::FORBIDDEN).body(Body::from("Access denied"));
    }

    match tokio::fs::read_dir(full_path).await {
        Ok(mut entries) => {
            let mut items = Vec::new();
            while let Ok(Some(entry)) = entries.next_entry().await {
                let metadata = entry.metadata().await.ok();
                let is_dir = metadata.map(|m| m.is_dir()).unwrap_or(false);
                let name = entry.file_name().to_string_lossy().to_string();
                
                // Calculate relative path for frontend usage
                // IMPORTANT: We strip the prefix so we return relative path "foo/bar", not "/src/media/foo/bar"
                let relative_path = entry.path()
                    .strip_prefix(&root)
                    .unwrap_or(&entry.path())
                    .to_string_lossy()
                    .to_string();

                items.push(FileEntry {
                    name,
                    is_dir,
                    path: relative_path, // Send RELATIVE path
                });
            }
            // Sort: directories first, then files
            items.sort_by(|a, b| {
                b.is_dir.cmp(&a.is_dir).then_with(|| a.name.cmp(&b.name))
            });
            
            Response::builder().status(StatusCode::OK).body(serde_json::to_string(&items).unwrap())
        }
        Err(e) => {
             Response::builder().status(StatusCode::INTERNAL_SERVER_ERROR).body(Body::from(e.to_string()))
        }
    }
}

// Proxy to VMSAM for folders list
#[handler]
async fn proxy_folders_list() -> impl IntoResponse {
    let client = reqwest::Client::new();
    let url = format!("{}/folders_list/", get_vmsam_host());
    
    match client.get(&url).send().await {
        Ok(resp) => {
             let status = resp.status();
             let body = resp.bytes().await.unwrap_or_default();
              Response::builder().status(status).body(Body::from(body))
        },
        Err(e) => {
            Response::builder().status(StatusCode::BAD_GATEWAY).body(Body::from(e.to_string()))
        }
    }
}

// Proxy for creating folder with absolute path enforcement
#[handler]
async fn proxy_create_folder(body: String) -> impl IntoResponse {
    let client = reqwest::Client::new();
    let url = format!("{}/folders/", get_vmsam_host());

    // Intercept and fix path
    let mut new_body = body.clone();
    if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&body) {
         if let Some(dest) = json.get("destination_path").and_then(|v| v.as_str()) {
             let root = get_create_root();
             // Ensure it's treated as relative
             let relative_dest = dest.trim_start_matches('/');
             let absolute_path = root.join(relative_dest);
             
             // Update JSON
             json["destination_path"] = serde_json::Value::String(absolute_path.to_string_lossy().to_string());
             if let Ok(s) = serde_json::to_string(&json) {
                 new_body = s;
             }
         }
    }

    match client.post(&url).header("Content-Type", "application/json").body(new_body).send().await {
         Ok(resp) => {
             let status = resp.status();
             let body = resp.bytes().await.unwrap_or_default();
              Response::builder().status(status).body(Body::from(body))
        },
        Err(e) => {
            Response::builder().status(StatusCode::BAD_GATEWAY).body(Body::from(e.to_string()))
        }
    }
}

// Proxy for creating regex
#[handler]
async fn proxy_create_regex(body: String) -> impl IntoResponse {
     let client = reqwest::Client::new();
    let url = format!("{}/regex/", get_vmsam_host());

    match client.post(&url).header("Content-Type", "application/json").body(body).send().await {
         Ok(resp) => {
             let status = resp.status();
             let body = resp.bytes().await.unwrap_or_default();
              Response::builder().status(status).body(Body::from(body))
        },
        Err(e) => {
            Response::builder().status(StatusCode::BAD_GATEWAY).body(Body::from(e.to_string()))
        }
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

pub fn routes() -> Route {
    Route::new()
        .at("/fs/list", poem::get(list_files))
        .at("/vmsam/folders_list", poem::get(proxy_folders_list))
        .at("/vmsam/folders", poem::post(proxy_create_folder))
        .at("/vmsam/regex", poem::post(proxy_create_regex))
}
