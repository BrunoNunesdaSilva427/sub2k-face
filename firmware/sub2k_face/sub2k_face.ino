
#include <avr/pgmspace.h>
#include "identities.h"

static const uint8_t SYNC_BYTE = 0xAA;
static const uint8_t VECTOR_LEN = 32;
static const uint8_t NAME_LEN = 12;


static const long THRESHOLD_EUCLIDEAN = 70;
static const long THRESHOLD_SQ = THRESHOLD_EUCLIDEAN * THRESHOLD_EUCLIDEAN;

static const uint8_t NUM_IDENTITIES = sizeof(IDENTITY_TABLE) / sizeof(IDENTITY_TABLE[0]);

int8_t incoming_vector[VECTOR_LEN];

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(3000);
}


long squared_distance_to_identity(const int8_t *query, uint8_t identity_idx) {
  long sum = 0;
  for (uint8_t i = 0; i < VECTOR_LEN; i++) {
    int8_t ref_val = (int8_t)pgm_read_byte(&IDENTITY_TABLE[identity_idx].vector[i]);
    long diff = (long)query[i] - (long)ref_val;
    sum += diff * diff;
  }
  return sum;
}

void send_response(bool matched, uint8_t identity_idx) {
  uint8_t response[1 + NAME_LEN];
  response[0] = matched ? 0x01 : 0x00;

  for (uint8_t i = 0; i < NAME_LEN; i++) {
    response[1 + i] = matched
      ? (uint8_t)pgm_read_byte(&IDENTITY_TABLE[identity_idx].name[i])
      : 0x00;
  }

  Serial.write(response, sizeof(response));
}

void loop() {

  if (Serial.available() == 0) return;
  if ((uint8_t)Serial.read() != SYNC_BYTE) return;

  size_t got = Serial.readBytes((char *)incoming_vector, VECTOR_LEN);
  if (got != VECTOR_LEN) {

    return;
  }

  if (NUM_IDENTITIES == 0) {

    send_response(false, 0);
    return;
  }

  uint8_t best_idx = 0;
  long best_dist_sq = squared_distance_to_identity(incoming_vector, 0);

  for (uint8_t i = 1; i < NUM_IDENTITIES; i++) {
    long dist_sq = squared_distance_to_identity(incoming_vector, i);
    if (dist_sq < best_dist_sq) {
      best_dist_sq = dist_sq;
      best_idx = i;
    }
  }

  bool matched = best_dist_sq <= THRESHOLD_SQ;
  send_response(matched, best_idx);
}
