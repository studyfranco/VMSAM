use anyhow::Result;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <file1> <file2> [pool_size]", args[0]);
        std::process::exit(2);
    }
    let f1 = &args[1];
    let f2 = &args[2];
    let pool = if args.len() >= 4 {
        args[3].parse::<usize>().unwrap_or(2)
    } else { 2 };

    let res = audio_sync::second_correlation_async(f1, f2, pool).await?;
    let json = serde_json::to_string(&res)?;
    println!("{}", json);
    Ok(())
}