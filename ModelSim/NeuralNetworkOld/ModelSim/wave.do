onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -label CLOCK_50 -radix binary /testbench/CLOCK_50
add wave -noupdate -label Reset -radix binary /testbench/reset

add wave -noupdate -divider Inputs
add wave -noupdate -label NNin -radix hexadecimal /testbench/NNin
add wave -noupdate -label NNvalid -radix hexadecimal /testbench/NNvalid

add wave -noupdate -divider Outputs
add wave -noupdate -label NNout -radix hexadecimal /testbench/NNout
add wave -noupdate -label NNoutValid -radix hexadecimal /testbench/NNoutValid
add wave -noupdate -label MaxNum -radix hexadecimal /testbench/maxIndex
add wave -noupdate -label MaxValue -radix hexadecimal /testbench/maxValue
add wave -noupdate -label MaxValid -radix hexadecimal /testbench/maxValid

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
WaveRestoreZoom {0 ps} {1000 ns}
