# for j, (k, v) in enumerate(sectors.items()):
#     print(k)
#     bitstrings_str = ffsim.addresses_to_strings(v, moldata.norb, moldata.nelec,
#         bitstring_type=ffsim.BitstringType.STRING, concatenate=False)
#     bitstrings_int = ffsim.addresses_to_strings(v, moldata.norb, moldata.nelec,
#         bitstring_type=ffsim.BitstringType.INT, concatenate=False)
#     for i in range(5):
#         w = bitstrings_int[0][i] ^ bitstrings_int[1][i]
#
#         print(bitstrings_str[0][i], bitstrings_str[1][i])
#         print(str(bin(w))[2:])
#     if j > 2:
#         break