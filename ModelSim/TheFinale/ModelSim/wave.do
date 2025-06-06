onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -label CLOCK_50 -radix binary /testbench/CLOCK_50
add wave -noupdate -label serialClock -radix binary /testbench/serialClock
add wave -noupdate -label Reset -radix binary /testbench/reset

add wave -noupdate -divider Inputs
add wave -noupdate -label data -radix hexadecimal /testbench/data
add wave -noupdate -label serialData -radix hexadecimal /testbench/serialData
add wave -noupdate -label KEY -radix binary /testbench/KEY
add wave -noupdate -label ARDUINO_IO -radix binary /testbench/ARDUINO_IO

add wave -noupdate -divider Internal
add wave -noupdate -label reset -radix hexadecimal /testbench/U1/reset

add wave -noupdate -divider "Neural Network"
add wave -noupdate -label NNreset -radix hexadecimal /testbench/U1/NNreset
add wave -noupdate -label NNin -radix hexadecimal /testbench/U1/NNin
add wave -noupdate -label NNout -radix hexadecimal /testbench/U1/NNout
add wave -noupdate -label NNvalid -radix hexadecimal /testbench/U1/NNvalid
add wave -noupdate -label NNoutValid -radix hexadecimal /testbench/U1/NNoutValid

add wave -noupdate -divider "HardMax"
add wave -noupdate -label maxIndex -radix hexadecimal /testbench/U1/maxIndex
add wave -noupdate -label maxValue -radix hexadecimal /testbench/U1/maxValue
add wave -noupdate -label maxValid -radix hexadecimal /testbench/U1/maxValid
add wave -noupdate -label maxIndex -radix hexadecimal /testbench/U1/maxIndex

add wave -noupdate -divider "Shift Register"
add wave -noupdate -label serialClock -radix hexadecimal /testbench/U1/serialClock
add wave -noupdate -label serialData -radix hexadecimal /testbench/U1/serialData
add wave -noupdate -label pushBuffer -radix hexadecimal /testbench/U1/pushBuffer

add wave -noupdate -divider Outputs
add wave -noupdate -label HEX0 -radix hexadecimal /testbench/HEX0


TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {100 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 80
configure wave -valuecolwidth 40
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {0 ps} {30000 ns}
