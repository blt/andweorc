#![no_std]

extern crate alloc;

// use addr2line::gimli::{EndianReader, RunTimeEndian};
// use procmaps::Mappings;
// use std::{collections::HashMap, fs, path::PathBuf, rc};

mod experiment;
mod per_thread;
mod posix;
pub mod progress_point;
mod timer;

// struct MapInfo {
//     base: usize,
//     // ceiling: usize,
//     offset: usize, // TODO possibly not useful
// }

// struct SharedObject {
//     load_address: u64,
//     context: addr2line::Context<EndianReader<RunTimeEndian, alloc::rc::Rc<[u8]>>>,
// }

// #[derive(Debug, Clone)]
// pub enum Location<'a> {
//     /// An instruction pointer that could not be resolved into a symbol
//     Raw(u64),
//     Symbol {
//         file: &'a str,
//         line: u32,
//     },
// }

// impl<'a> Default for Location<'a> {
//     fn default() -> Self {
//         Location::Raw(0)
//     }
// }

// /// Performs resolution of an instruction pointer to a symbol, line number et al
// pub struct SymbolResolver {
//     contexts: HashMap<PathBuf, SharedObject>,
// }

// impl SymbolResolver {
//     pub fn new() -> Self {
//         let pid = std::process::id();
//         let mut mmapped_info: HashMap<PathBuf, MapInfo> = HashMap::new();
//         for (p, mi) in Mappings::from_pid(pid as i32)
//             .unwrap()
//             .iter()
//             .filter(|m| m.perms.executable)
//             .filter(|m| matches!(m.pathname, procmaps::Path::MappedFile(..)))
//             .map(|m| {
//                 let path = match m.pathname {
//                     procmaps::Path::MappedFile(ref p) => PathBuf::from(p),
//                     _ => unreachable!(),
//                 };

//                 (
//                     path,
//                     MapInfo {
//                         base: m.base,
//                         // ceiling: m.ceiling,
//                         offset: m.offset,
//                     },
//                 )
//             })
//         {
//             mmapped_info.insert(p, mi);
//         }

//         let mut contexts: HashMap<PathBuf, _> = HashMap::new();
//         for (p, mi) in mmapped_info.into_iter() {
//             let file = fs::File::open(&p).unwrap();
//             let map = unsafe { memmap::Mmap::map(&file).unwrap() };
//             let obj = addr2line::object::read::File::parse(&*map).unwrap();
//             let ctx = addr2line::Context::new(&obj).unwrap();
//             let so = SharedObject {
//                 load_address: (mi.base - mi.offset) as u64,
//                 context: ctx,
//             };
//             contexts.insert(p, so);
//         }

//         Self { contexts }
//     }

//     // pub fn enrich<'a>(&'a self, addrs: &[u64], buf: &mut Vec<Location<'a>>) -> Result<(), Error> {
//     //     for addr in addrs {
//     //         let location = if let Some(loc) = self.find(*addr)? {
//     //             match (loc.file, loc.line) {
//     //                 (Some(file), Some(line)) => Location::Symbol { file, line },
//     //                 _ => Location::Raw(*addr),
//     //             }
//     //         } else {
//     //             Location::Raw(*addr)
//     //         };
//     //         buf.push(location);
//     //     }
//     //     Ok(())
//     // }

//     pub fn find(&self, addr: u64) -> Result<Option<addr2line::Location<'_>>, Error> {
//         for (_path, ctx) in self.contexts.iter() {
//             match ctx.context.find_location(addr - ctx.load_address) {
//                 Ok(Some(loc)) => {
//                     return Ok(Some(loc));
//                 }
//                 _ => continue,
//             }
//         }
//         Ok(None)
//     }
// }
