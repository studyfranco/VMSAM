use poem::{
    endpoint::StaticFilesEndpoint, listener::TcpListener, middleware::Cors, EndpointExt, Route, Server,
};
use std::env;

mod api;

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "poem=debug");
    }
    tracing_subscriber::fmt::init();

    let app = Route::new()
        .nest("/api", api::routes())
        .nest(
            "/",
            StaticFilesEndpoint::new("static").index_file("index.html"),
        )
        .with(Cors::new());

    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);
    println!("Server running at http://{}", addr);

    Server::new(TcpListener::bind(addr))
        .run(app)
        .await
}
