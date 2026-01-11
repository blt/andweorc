//! Procedural macros for the andweorc causal profiler.
//!
//! This crate provides the `#[profile]` attribute macro that instruments
//! functions for causal profiling.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Instruments a function for causal profiling.
///
/// This macro wraps the function body to:
/// 1. Record a progress point when the function completes
/// 2. Optionally inject delay points for virtual speedup experiments
///
/// # Example
///
/// ```ignore
/// use andweorc_macros::profile;
///
/// #[profile]
/// fn process_request(req: Request) -> Response {
///     // ... process the request ...
/// }
/// ```
///
/// The above expands to something like:
///
/// ```ignore
/// fn process_request(req: Request) -> Response {
///     let __result = {
///         // ... process the request ...
///     };
///     andweorc::progress!("process_request");
///     __result
/// }
/// ```
#[proc_macro_attribute]
pub fn profile(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    // Parse optional progress point name from attribute
    let progress_name = if attr.is_empty() {
        // Use function name as default progress point name
        input.sig.ident.to_string()
    } else {
        // Parse the provided name
        let name_lit: syn::LitStr = parse_macro_input!(attr as syn::LitStr);
        name_lit.value()
    };

    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;
    let attrs = &input.attrs;

    // Generate the instrumented function
    // We call the progress! macro which is already exported from andweorc
    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            let __andweorc_result = { #block };
            andweorc::progress!(#progress_name);
            __andweorc_result
        }
    };

    TokenStream::from(expanded)
}

/// Marks a location as a delay injection point.
///
/// During causal profiling experiments, this macro will inject delays
/// to simulate virtual speedups of other code locations.
///
/// # Example
///
/// ```ignore
/// use andweorc_macros::delay_point;
///
/// fn compute() {
///     for item in items {
///         process(item);
///         delay_point!(); // Delay injection happens here
///     }
/// }
/// ```
#[proc_macro]
pub fn delay_point(_input: TokenStream) -> TokenStream {
    let expanded = quote! {
        {
            // Get the current thread's profiler and inject delay if needed
            let __tid = ::nix::sys::pthread::pthread_self();
            if let Some(__exp) = ::andweorc::experiment::EXPERIMENT.get() {
                if let Some(mut __ts) = __exp.thread_states.get_mut(&__tid) {
                    // Note: delay() requires &mut self, need interior mutability
                    // For now this is a no-op placeholder
                }
            }
        }
    };

    TokenStream::from(expanded)
}
