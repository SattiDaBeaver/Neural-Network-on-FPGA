onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -label CLOCK_50 -radix binary /testbench/CLOCK_50
add wave -noupdate -label Reset -radix binary /testbench/reset

add wave -noupdate -divider Inputs
add wave -noupdate -label layerIn -radix hexadecimal /testbench/layerIn
add wave -noupdate -label layerValid -radix hexadecimal /testbench/layerValid

add wave -noupdate -divider "Internal Nets"

add wave -noupdate -divider Outputs
add wave -noupdate -label layerOut -radix hexadecimal /testbench/layerOut
add wave -noupdate -label layerOutValid -radix hexadecimal /testbench/layerOutValid

add wave -noupdate -label counterOut -radix hexadecimal /testbench/inputCounterOut
add wave -noupdate -label counterValid -radix hexadecimal /testbench/inputCounterValid
add wave -noupdate -label serializerOut -radix hexadecimal /testbench/serializerOut

TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {10000 ps} 0}
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
WaveRestoreZoom {0 ps} {300 ns}
