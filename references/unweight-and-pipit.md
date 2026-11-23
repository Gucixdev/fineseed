# project_pipit / Cloudflare (partly unverified)

Verified: Cloudflare's own blog post "Unweight: how we compressed an LLM 22% without sacrificing quality"
(blog.cloudflare.com/unweight-tensor-compression): Huffman coding of the redundant exponent bytes of BF16 MLP weights,
about 13% (inference bundles, gate/up projections) up to 22% (distribution bundles, all MLP projections), bit exact.
NOT verified: "Project Pipit" with a 5.2x claim exists only in two secondary articles, no Cloudflare page or repository.
Treat the number as a rumour; ask him what he means by project_pipit.

Take: confirms that Huffman on the exponent plane is the core; nothing else needed from it.
