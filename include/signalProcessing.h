#include <Arduino.h>

bool checkBeat(int32_t sample);
int16_t DCfilter(int32_t *p, int16_t x);
int16_t lowPassFIRfilter(int16_t din);
int32_t mul16(int16_t x, int16_t y);