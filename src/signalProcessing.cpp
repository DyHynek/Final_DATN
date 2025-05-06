#include <signalProcessing.h>

int16_t IR_AC_max = 20;
int16_t IR_AC_min = -20;

int16_t IR_AC_signal_current = 0;
int16_t IR_AC_signal_previous;
int16_t IR_AC_signal_min = 0;
int16_t IR_AC_signal_max = 0;
int16_t IR_average_estimated;

int16_t positiveEdge = 0;
int16_t negativeEdge = 0;
int32_t ir_avg_reg =0;

int16_t cbuf[32];
uint8_t offset = 0;

//hệ số bộ lọc FIR thông thấp
static const uint16_t FIRcoeffs[12] = {172, 321, 579, 927, 1360, 1858, 2390, 2916, 3391, 3768, 4012, 4096};

bool checkBeat(int32_t sample)
{
    bool beatDetected = false;

    IR_AC_signal_previous = IR_AC_signal_current;

    IR_average_estimated = DCfilter(&ir_avg_reg, sample);
    // IR_AC_signal_current = lowPassFIRfilter(sample - IR_average_estimated);
    IR_AC_signal_current = sample - IR_average_estimated;
    if((IR_AC_signal_previous < 0) && (IR_AC_signal_current >= 0))
    {
        IR_AC_max = IR_AC_signal_max;
        IR_AC_min = IR_AC_signal_min;
        
        positiveEdge = 1;
        negativeEdge = 0;
        IR_AC_signal_max = 0;

        if((IR_AC_max - IR_AC_min) > 20 && (IR_AC_max - IR_AC_min) < 1000)
        {
            beatDetected = true;
            //Serial.println(IR_AC_signal_current);
        }

    }

    if((IR_AC_signal_previous > 0) && (IR_AC_signal_current <= 0))
    {
        positiveEdge = 0;
        negativeEdge = 1;
        IR_AC_signal_min = 0;
    }

    if(positiveEdge & (IR_AC_signal_current > IR_AC_signal_previous))
    {
        IR_AC_signal_max = IR_AC_signal_current;
    }

    if(negativeEdge & (IR_AC_signal_current < IR_AC_signal_previous))
    {
        IR_AC_signal_min = IR_AC_signal_current;
    }

    return (beatDetected);
}

//hàm lọc thành phần DC
int16_t DCfilter(int32_t *p, int16_t x)
{
    *p += ((((long) x << 15) - *p) >> 4);
    return (*p >> 15);
}

//hàm lọc FIR thông thấp
int16_t lowPassFIRfilter(int16_t din)
{
    cbuf[offset] = din;

    int32_t z = mul16(FIRcoeffs[11], cbuf[(offset -11) & 0x1F]);

    for(uint8_t i = 0; i < 11; i++)
    {
        z += mul16(FIRcoeffs[i], cbuf[(offset - i) & 0x1F] + cbuf[(offset - 22 + i) & 0x1F]);
    }

    offset++;
    offset %= 32;
    
    return(z >> 15);
}

int32_t mul16(int16_t x, int16_t y)
{
    return((long)x * (long)y);
}