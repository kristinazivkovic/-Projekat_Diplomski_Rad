import ssl

found_any = False

for storename in ("CA", "ROOT"):
    print(f"\n--- Provera store-a: {storename} ---")
    for der_bytes, encoding, trust in ssl.enum_certificates(storename):
        try:
            pem = ssl.DER_cert_to_PEM_cert(der_bytes)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.load_verify_locations(cadata=pem)
        except ssl.SSLError as e:
            found_any = True
            print("NEISPRAVAN SERTIFIKAT NAĐEN:")
            print("  Trust:", trust)
            print("  Greška:", e)
            print("  Broj bajtova (DER):", len(der_bytes))

if not found_any:
    print("\nNijedan pojedinačan sertifikat nije pukao.")
    print("Mozda problem nastaje tek kad se ucitaju SVI odjednom (npr. duplikat ili poredak).")

print("\nGotovo.")