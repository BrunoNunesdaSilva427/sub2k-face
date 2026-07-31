

import argparse
import json

NAME_LEN = 12
VECTOR_LEN = 32


def format_identity(name, vector):
    if len(vector) != VECTOR_LEN:
        raise ValueError(
            f"Identidade '{name}': vetor tem {len(vector)} valores, esperado {VECTOR_LEN}."
        )
    for v in vector:
        if not -127 <= v <= 127:
            raise ValueError(
                f"Identidade '{name}': valor {v} fora da faixa int8 (-127..127)."
            )

    name_bytes = name.encode("ascii")
    if len(name_bytes) > NAME_LEN - 1:
        raise ValueError(
            f"Identidade '{name}': nome tem mais de {NAME_LEN - 1} caracteres ASCII "
            "(limite da tabela no Arduino)."
        )

    padded = name_bytes + b"\x00" * (NAME_LEN - len(name_bytes))
    name_literal = ", ".join(str(b) for b in padded)
    vector_literal = ", ".join(str(int(v)) for v in vector)

    return f"  {{ {{ {name_literal} }}, {{ {vector_literal} }} }},  // {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", default="identities.json")
    ap.add_argument("--out", default="firmware/identities.h")
    args = ap.parse_args()

    with open(args.identities) as f:
        identities = json.load(f)

    if not identities:
        raise SystemExit(
            "identities.json está vazio -- cadastre pelo menos uma identidade "
            "com enroll.py antes de gerar o header."
        )

    entries = [format_identity(name, vector) for name, vector in identities.items()]

    header = f"""#ifndef IDENTITIES_H
#define IDENTITIES_H

#include <avr/pgmspace.h>
#include <stdint.h>

struct Identity {{
  char name[12];
  int8_t vector[32];
}};

// Gerado automaticamente por tools/generate_identities_header.py a partir
// de {args.identities} -- NAO EDITE A MAO, regenere o arquivo em vez disso.
// {len(entries)} identidade(s): {", ".join(identities.keys())}
const Identity IDENTITY_TABLE[] PROGMEM = {{
{chr(10).join(entries)}
}};

#endif
"""

    with open(args.out, "w") as f:
        f.write(header)

    print(
        f"{len(entries)} identidade(s) escritas em {args.out}. "
        "Copie esse arquivo pra pasta do sketch (junto do .ino) e regrave o Arduino."
    )


if __name__ == "__main__":
    main()
