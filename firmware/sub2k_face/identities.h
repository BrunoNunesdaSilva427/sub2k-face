#ifndef IDENTITIES_H
#define IDENTITIES_H

#include <avr/pgmspace.h>
#include <stdint.h>

struct Identity {
  char name[12];
  int8_t vector[32];
};

const Identity IDENTITY_TABLE[] PROGMEM = {

};

#endif
