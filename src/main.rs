//! fineseed pack IN [-b BASE] [-l LEVEL] [-o OUT] | unpack IN [-b BASE] [-o OUT] | info IN | check IN [-b BASE]
use fineseed::archive;
use fineseed::Res;
use std::path::PathBuf;

const HELP: &str = "fineseed - lossless .zip for AI models (safetensors, GGUF, ONNX, anything else by what the bytes look like)
  fineseed pack IN... [-l LEVEL] [-o OUT.fseed]   files or folders into one archive; OUT defaults to IN.fseed, LEVEL (1-22) to 3
                                                  identical blocks are stored once, a fine-tune against the most similar
                                                  earlier tensor; a .fseed among the inputs is opened and its members
                                                  share all of that (nesting, built again byte for byte on unpack)
  fineseed unpack IN.fseed [-o DIR]               every file under DIR (default .), each checked against its blake3
  fineseed check IN.fseed                         unpack to a scratch folder next to IN.fseed, verify, delete it
  fineseed list IN.fseed                          every entry, nested ones indented";

fn show(label: &str, r: &archive::Report) {
    let pct = if r.original == 0 {
        100.0
    } else {
        100.0 * r.packed as f64 / r.original as f64
    };
    let nested = if r.nested > 0 {
        format!(", {} nested", r.nested)
    } else {
        String::new()
    };
    println!(
        "{label}: {} files{nested}, {} -> {} bytes ({pct:.1}%), {} blocks, {} copies of earlier blocks, {} against a base, {:.1}s",
        r.members, r.original, r.packed, r.blocks, r.dup_blocks, r.xor_blocks, r.seconds
    );
}

fn run(args: &[String]) -> Res<()> {
    let cmd = args.first().map(String::as_str).ok_or(HELP)?;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let (mut out, mut level) = (None, 3i32);
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" => out = Some(PathBuf::from(it.next().ok_or("-o needs a path")?)),
            "-l" => {
                level = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("-l needs a number")?
            }
            "-h" | "--help" => return Err(HELP.into()),
            p if p.starts_with('-') && p.len() > 1 => return Err(format!("unknown option {p}")),
            p => inputs.push(PathBuf::from(p)),
        }
    }
    let first = inputs.first().cloned().ok_or(HELP)?;
    match cmd {
        "pack" => {
            let out = match out {
                Some(o) => o,
                None if inputs.len() == 1 => {
                    let mut n = first.as_os_str().to_owned();
                    n.push(".fseed");
                    PathBuf::from(n)
                }
                None => return Err("pack of several inputs needs -o OUT.fseed".into()),
            };
            show("packed", &archive::store(&archive::collect(&inputs)?, &out, level)?);
        }
        "unpack" if inputs.len() == 1 => show(
            "unpacked",
            &archive::restore(&first, &out.unwrap_or_else(|| PathBuf::from(".")))?,
        ),
        "check" if inputs.len() == 1 => {
            let mut scratch = first.as_os_str().to_owned();
            scratch.push(".check");
            let scratch = PathBuf::from(scratch);
            let r = archive::restore(&first, &scratch);
            let _ = std::fs::remove_dir_all(&scratch);
            show("verified", &r?);
        }
        "list" if inputs.len() == 1 => {
            for m in archive::list(&first)? {
                let pad = "  ".repeat(m.depth as usize);
                let tag = match m.nest {
                    Some((l, k)) => format!("  [fseed, level {l}, {k} entries]"),
                    None => String::new(),
                };
                println!("{:>14}  {pad}{}{tag}", m.len, m.name);
            }
        }
        _ => return Err(HELP.into()),
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
