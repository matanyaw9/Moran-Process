"""Constants shared by readers and writers of the CSV artefacts."""


#: Hash columns must never be type-inferred. A Weisfeiler-Lehman hash is 32 hex
#: characters, and roughly one in a few thousand of them matches the shape of a float in
#: scientific notation -- ``420e50584341715888a2c9b067b56b98`` parses as 420 x 10^50584...
#: pandas then SEGFAULTS on the exponent, killing the process outright: not an exception,
#: so nothing can catch or retry it. Reproduced with a one-line CSV containing only that
#: hash. Passing an explicit dtype skips the inference entirely.
HASH_DTYPES = {"wl_hash": "string", "parent_wl_hash": "string"}
