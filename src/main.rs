use anyhow::Result;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <file1> <file2>", args[0]);
        std::process::exit(2);
    }
    let res = audio_sync::second_correlation(&args[1], &args[2])?;
    println!("{}", serde_json::to_string(&res)?);
    Ok(())
}
