//! fineseed pack | unpack | check | list | cat | extract | mount | protect | scrub | repair, see HELP
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
  fineseed list IN.fseed                          every entry, nested ones indented
  fineseed cat IN.fseed NAME                      one entry to standard output, nothing else is restored
  fineseed extract IN.fseed [NAME...] [-o DIR]    some entries (a folder or a nested archive takes all under it), each checked
  fineseed mount IN.fseed DIR [--cache MIB]       the archive as a read-only folder; reads decode only what they touch
                                                  (fusermount3 -u DIR to let go)
  fineseed protect IN.fseed [--parity PCT]        Reed-Solomon parity behind the archive (default 5 %), so rot can be repaired;
                                                  pack takes --parity PCT too. Readers ignore the parity.
  fineseed scrub IN.fseed                         check every 64 KiB against its CRC32, change nothing
  fineseed repair IN.fseed                        rebuild damaged 64 KiB pieces from the parity, in place
  pack with `-` as input reads standard input (one member, named stdin, or --name N), with `-o -` writes the archive to
  standard output; --dtype bf16|f16|f32|i16|i32|i64|u8 says what a stream is, else each 4 MiB of it is guessed";

fn show(label: &str, r: &archive::Report, to_stderr: bool) {
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
    let line = format!(
        "{label}: {} files{nested}, {} -> {} bytes ({pct:.1}%), {} blocks, {} copies of earlier blocks, {} against a base, {:.1}s",
        r.members, r.original, r.packed, r.blocks, r.dup_blocks, r.xor_blocks, r.seconds
    );
    if to_stderr {
        eprintln!("{line}");
    } else {
        println!("{line}");
    }
}

fn run(args: &[String]) -> Res<()> {
    let cmd = args.first().map(String::as_str).ok_or(HELP)?;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let (mut out, mut level) = (None, 3i32);
    let (mut dtype, mut name, mut cache) = (None, None::<String>, 512usize);
    let mut parity: Option<u32> = None;
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
            "--dtype" => {
                dtype = Some(match it.next().map(String::as_str) {
                    Some("bf16") => fineseed::split::BF16,
                    Some("f16") => fineseed::split::F16,
                    Some("f32") => fineseed::split::F32,
                    Some("i16") => 2,
                    Some("i32") => 4,
                    Some("i64") => 8,
                    Some("u8") => fineseed::split::PLAIN,
                    _ => return Err("--dtype is one of bf16 f16 f32 i16 i32 i64 u8".into()),
                })
            }
            "--parity" => {
                parity = Some(
                    it.next()
                        .and_then(|v| v.parse().ok())
                        .ok_or("--parity needs a percentage, 1 to 400")?,
                )
            }
            "--name" => name = Some(it.next().ok_or("--name needs a name")?.clone()),
            "--cache" => {
                cache = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--cache needs a number of MiB")?
            }
            "-h" | "--help" => return Err(HELP.into()),
            "-" => inputs.push(PathBuf::from("-")),
            p if p.starts_with('-') && p.len() > 1 => return Err(format!("unknown option {p}")),
            p => inputs.push(PathBuf::from(p)),
        }
    }
    let first = inputs.first().cloned().ok_or(HELP)?;
    // an archive written to standard output must keep the report off it
    let to_stdout = out.as_deref() == Some(std::path::Path::new("-"));
    match cmd {
        "pack" => {
            let out = match out {
                Some(o) => o,
                None if inputs.len() == 1 && first != std::path::Path::new("-") => {
                    let mut n = first.as_os_str().to_owned();
                    n.push(".fseed");
                    PathBuf::from(n)
                }
                None => return Err("pack of several inputs, or of standard input, needs -o OUT.fseed".into()),
            };
            let mut files = archive::collect(&inputs)?;
            if let Some(n) = &name {
                for f in files.iter_mut().filter(|f| f.1 == std::path::Path::new("-")) {
                    f.0 = n.clone();
                }
            }
            let r = archive::store_as(&files, &out, level, dtype)?;
            show("packed", &r, to_stdout);
            if let Some(pct) = parity {
                if to_stdout {
                    return Err("--parity needs a file to write to, not standard output".into());
                }
                let n = fineseed::parity::protect(&out, None, pct)?;
                eprintln!("parity: {pct} %, {n} bytes added");
            }
        }
        "protect" if inputs.len() == 1 => {
            let n = fineseed::parity::protect(&first, out.as_deref(), parity.unwrap_or(5))?;
            println!("parity added: {n} bytes");
        }
        "scrub" | "repair" if inputs.len() == 1 => {
            let fix = cmd == "repair";
            let h = if fix {
                fineseed::parity::repair(&first)?
            } else {
                fineseed::parity::verify(&first)?
            };
            println!(
                "{} pieces + {} parity pieces: {} damaged, {} parity damaged, {} rebuilt, {} groups beyond repair",
                h.shards,
                h.recovery,
                h.bad.len(),
                h.bad_recovery.len(),
                h.fixed,
                h.lost_groups.len()
            );
            if !h.lost_groups.is_empty() || (!fix && !h.clean()) {
                return Err(if fix { "damage beyond what the parity can repair" } else { "damaged: run repair" }.into());
            }
        }
        "unpack" if inputs.len() == 1 => show(
            "unpacked",
            &archive::restore(&first, &out.unwrap_or_else(|| PathBuf::from(".")))?,
            false,
        ),
        "cat" if inputs.len() == 2 => {
            let ar = archive::access::Archive::open(&first, 64 << 20)?;
            let want = inputs[1].to_string_lossy();
            let i = ar.find(&want).ok_or(format!("no entry called {want}"))?;
            ar.write_member(i, &mut std::io::stdout().lock())?;
        }
        "extract" if !inputs.is_empty() => {
            let ar = archive::access::Archive::open(&first, 64 << 20)?;
            let names: Vec<String> = inputs[1..].iter().map(|p| p.to_string_lossy().into_owned()).collect();
            let n = ar.extract(&names, &out.unwrap_or_else(|| PathBuf::from(".")))?;
            println!("extracted {n} files");
        }
        #[cfg(feature = "mount")]
        "mount" if inputs.len() == 2 => fineseed::mount::mount(&first, &inputs[1], cache)?,
        "check" if inputs.len() == 1 => {
            let mut scratch = first.as_os_str().to_owned();
            scratch.push(".check");
            let scratch = PathBuf::from(scratch);
            let r = archive::restore(&first, &scratch);
            let _ = std::fs::remove_dir_all(&scratch);
            show("verified", &r?, false);
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
        // `fineseed cat ... | head` closes the pipe on purpose
        if e.contains("Broken pipe") {
            return;
        }
        eprintln!("{e}");
        std::process::exit(1);
    }
}
