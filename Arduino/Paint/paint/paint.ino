// the regular Adafruit "TouchScreen.h" library only works on AVRs

// different mcufriend shields have Touchscreen on different pins
// and rotation.
// Run the TouchScreen_Calibr_native sketch for calibration of your shield

#include <MCUFRIEND_kbv.h>
MCUFRIEND_kbv tft;       // hard-wired for UNO shields anyway.
#include <TouchScreen.h>

const int XP=8,XM=A2,YP=A3,YM=9; //320x480 ID=0x9486
// Calibration
const int TS_LEFT=130,TS_RT=903,TS_TOP=954,TS_BOT=95;

TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);
TSPoint tp;

#define MINPRESSURE 200
#define MIDPRESSURE 550
#define MAXPRESSURE 1200
// #define REVERSE_INPUT

int16_t BOXSIZE;
int16_t PENRADIUS = 1;
uint16_t ID, oldcolor, currentcolor;
uint8_t Orientation = 0;    //PORTRAIT

// Assign human-readable names to some common 16-bit color values:
#define BLACK       0x0000
#define BLUE        0x001F
#define RED         0xF800
#define GREEN       0x07E0
#define CYAN        0x07FF
#define MAGENTA     0xF81F
#define YELLOW      0xFFE0
#define GREY        0xC618
#define WHITE       0xFFFF
#define GREENYELLOW 0xAFE5
#define DARKGREY    0x7BEF

// Pins to Shift Out Data
#define SHIFT_CLK       0 
#define SHIFT_DATA      1
#define SHIFT_ENABLE    A0

// MNIST and Display Constants
#define MNIST_PIXELS 784 // 28x28 pixels
#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28

#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 480

#define X_PADDING 6
#define Y_PADDING 20
#define PIXEL_SIZE 11 // Size of each pixel in the display

#define CLEAR_HEIGHT 60
#define PUSH_HEIGHT 60

#define NUM_BYTES 98 // 98 * 8 = 784

// Internal Buffer
uint8_t buffer[NUM_BYTES]; // 28x28 pixels, 1 bit per pixel
// uint8_t pixelSize = SCREEN_WIDTH / MNIST_WIDTH; // Size of each pixel in the display

// Global Variables
uint16_t xpos, ypos;  //screen coordinates
bool bufferChanged = false; // Flag to track if buffer has changed

// Function Prototypes
void TFTsetup(void);
void initializeBuffer(void);
void drawBuffer(void);
void drawFullBuffer(void);
void getPosition(void);
void clearScreenBuffer(void);
void setPixelBit(uint16_t index);
void drawPixelMNIST(void);
void drawSmallPixelMNIST(void);
void shiftBuffer(void);


void setup(void){
    pinMode(SHIFT_CLK, OUTPUT);
    pinMode(SHIFT_DATA, OUTPUT);
    pinMode(SHIFT_ENABLE, OUTPUT);

    TFTsetup(); // Initialize the TFT display
    clearScreenBuffer();
    drawBuffer(); // Draw the initial buffer on the display
}

void loop(){
    for (int i = 0; i < 30; i++){
        for (int i = 0; i < 10; i++){
            tp = ts.getPoint();   //tp.x, tp.y are ADC values

            // if sharing pins, you'll need to fix the directions of the touchscreen pins
            pinMode(XM, OUTPUT);
            pinMode(YP, OUTPUT);
            // we have some minimum pressure we consider 'valid'
            // pressure of 0 means no pressing!

            if (tp.z > MINPRESSURE && tp.z < MAXPRESSURE) {
                getPosition();

                // are we in drawing area ?
                if ((ypos > Y_PADDING) && (ypos < (Y_PADDING + PIXEL_SIZE * MNIST_HEIGHT))) {
                    if (tp.z < MIDPRESSURE){
                        drawSmallPixelMNIST();
                    }
                    else {
                        drawSmallPixelMNIST();
                    }
                    bufferChanged = true;
                }

                // Push Buffer Button
                if ((ypos > SCREEN_HEIGHT - PUSH_HEIGHT - CLEAR_HEIGHT) && (ypos < SCREEN_HEIGHT - CLEAR_HEIGHT)) {
                    tft.setCursor(30, 200);
                    tft.setTextColor(GREY, RED);
                    tft.setTextSize(3);
                    tft.print("Pushing Buffer");
                    shiftBuffer();
                    drawFullBuffer();
                }

                // Clear Screen Button
                if (ypos > SCREEN_HEIGHT - CLEAR_HEIGHT) {
                    bufferChanged = true;
                    clearScreenBuffer();
                }
            }
        }
        drawBuffer();
    }
    if (bufferChanged) {
        bufferChanged = false;
        shiftBuffer();
    }
    
}

// Function Definitions
void TFTsetup(void){
    tft.reset();
    ID = tft.readID();
    tft.begin(ID);
    tft.setRotation(Orientation);
    tft.fillScreen(BLACK);
}

void initializeBuffer(void) {
    for (int i = 0; i < NUM_BYTES; i++) {
        buffer[i] = 0x00;
    }
}

void drawBuffer(void) {
    for (int y = 0; y < MNIST_HEIGHT; y++) {
        for (int x = 0; x < MNIST_WIDTH; x++) {
            uint16_t index = y * MNIST_WIDTH + x;
            uint8_t byteIndex = index / 8;
            uint8_t bitIndex  = index % 8;

            uint16_t color = ((buffer[byteIndex] >> bitIndex) & 1) ? WHITE : BLACK;

            uint16_t xScreen = x * PIXEL_SIZE + X_PADDING;
            uint16_t yScreen = y * PIXEL_SIZE + Y_PADDING; 
            if ((color != BLACK) && (tft.readPixel(xScreen + 1, yScreen + 1) == BLACK)){
                tft.fillRect(xScreen, yScreen, PIXEL_SIZE, PIXEL_SIZE, color);
            }
        }
    }
}

void drawFullBuffer(void){
    for (int y = 0; y < MNIST_HEIGHT; y++) {
        for (int x = 0; x < MNIST_WIDTH; x++) {
            uint16_t index = y * MNIST_WIDTH + x;
            uint8_t byteIndex = index / 8;
            uint8_t bitIndex  = index % 8;

            uint16_t color = ((buffer[byteIndex] >> bitIndex) & 1) ? WHITE : BLACK;

            uint16_t xScreen = x * PIXEL_SIZE + X_PADDING;
            uint16_t yScreen = y * PIXEL_SIZE + Y_PADDING; 
            tft.fillRect(xScreen, yScreen, PIXEL_SIZE, PIXEL_SIZE, color);
        }
    }
}

void getPosition(void){
    switch (Orientation) {
        case 0:
            xpos = map(tp.x, TS_LEFT, TS_RT, 0, tft.width());
            ypos = map(tp.y, TS_TOP, TS_BOT, 0, tft.height());
            break;
        case 1:
            xpos = map(tp.y, TS_TOP, TS_BOT, 0, tft.width());
            ypos = map(tp.x, TS_RT, TS_LEFT, 0, tft.height());
            break;
        case 2:
            xpos = map(tp.x, TS_RT, TS_LEFT, 0, tft.width());
            ypos = map(tp.y, TS_BOT, TS_TOP, 0, tft.height());
            break;
        case 3:
            xpos = map(tp.y, TS_BOT, TS_TOP, 0, tft.width());
            ypos = map(tp.x, TS_LEFT, TS_RT, 0, tft.height());
            break;
    }
}

void clearScreenBuffer(void){
    initializeBuffer();
    tft.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GREY);
    tft.fillRect(X_PADDING, Y_PADDING, MNIST_WIDTH * PIXEL_SIZE, MNIST_HEIGHT * PIXEL_SIZE, BLACK);

    // Push Buffer Button
    tft.fillRect(0, SCREEN_HEIGHT - CLEAR_HEIGHT - PUSH_HEIGHT, SCREEN_WIDTH, PUSH_HEIGHT, YELLOW);
    tft.setCursor(28, SCREEN_HEIGHT - CLEAR_HEIGHT - PUSH_HEIGHT + 15);
    tft.setTextColor(BLACK);
    tft.setTextSize(4);
    tft.print("Push Buffer");

    // Clear Screen Button
    tft.fillRect(0, SCREEN_HEIGHT - CLEAR_HEIGHT, SCREEN_WIDTH, CLEAR_HEIGHT, CYAN);
    tft.setCursor(17, SCREEN_HEIGHT - CLEAR_HEIGHT + 15);
    tft.setTextColor(BLACK);
    tft.setTextSize(4);
    tft.print("Clear Screen");

}

// Helper to set a bit in the buffer
void setPixelBit(uint16_t index) {
    if (index >= MNIST_PIXELS) return;
    uint8_t byteIndex = index / 8;
    uint8_t bitIndex  = index % 8;
    buffer[byteIndex] |= (1 << bitIndex);
}

void drawPixelMNIST(void) {
    uint8_t xbuf = (xpos - X_PADDING) / PIXEL_SIZE;
    uint8_t ybuf = (ypos - Y_PADDING) / PIXEL_SIZE;
    
    // Pixel indices in flat [0..783] buffer
    uint16_t pixel[5] = {
        ybuf * MNIST_WIDTH + xbuf,         // Center
        (ybuf - 1) * MNIST_WIDTH + xbuf,   // Top
        (ybuf + 1) * MNIST_WIDTH + xbuf,   // Bottom
        ybuf * MNIST_WIDTH + (xbuf - 1),   // Left
        ybuf * MNIST_WIDTH + (xbuf + 1)    // Right
    };

    for (int i = 0; i < 5; i++) {
        // Bounds check
        if (pixel[i] < MNIST_PIXELS) {
            // Prevent horizontal wrap from left/right neighbors
            if ((i == 3 || i == 4) && (pixel[0] / MNIST_WIDTH) != (pixel[i] / MNIST_WIDTH)) {
                continue;
            }
            setPixelBit(pixel[i]);
        }
    }
}

void drawSmallPixelMNIST(void){
    uint8_t xbuf = (xpos - X_PADDING) / PIXEL_SIZE;
    uint8_t ybuf = (ypos - Y_PADDING) / PIXEL_SIZE;

    setPixelBit(ybuf * MNIST_WIDTH + xbuf);
}

void shiftBuffer(void) {
#ifdef REVERSE_INPUT
    // MSB-first across bitstream
    for (int byteIdx = 0; byteIdx < 98; byteIdx++) {
        uint8_t b = 0;
        for (int i = 0; i < 8; i++) {
            int bitIdx = 783 - (byteIdx * 8 + i);
            if (bitIdx < 0) break;

            int srcByte = bitIdx / 8;
            int srcBit = bitIdx % 8;
            uint8_t bit = (buffer[srcByte] >> srcBit) & 0x1;
            b |= (bit << (7 - i));
        }
        shiftOut(SHIFT_DATA, SHIFT_CLK, MSBFIRST, b);
    }
#else
    // LSB-first across bitstream
    for (int byteIdx = 0; byteIdx < 98; byteIdx++) {
        shiftOut(SHIFT_DATA, SHIFT_CLK, LSBFIRST, buffer[byteIdx]);
    }
#endif
}






