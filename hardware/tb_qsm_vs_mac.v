`timescale 1ns/1ps

module tb_qsm_vs_mac;

    reg clk;
    reg reset;
    reg en;
    reg [7:0] w_val;
    reg [7:0] x_val;

    wire [31:0] mac_acc;
    wire [31:0] qsm_acc;

    // Instantiate MAC
    mac_unit u_mac (
        .clk(clk),
        .reset(reset),
        .en(en),
        .w_val(w_val),
        .x_val(x_val),
        .acc_out(mac_acc)
    );

    // Instantiate QSM
    qsm_unit u_qsm (
        .clk(clk),
        .reset(reset),
        .en(en),
        .w_val(w_val),
        .x_val(x_val),
        .acc_out(qsm_acc)
    );

    // Clock generator
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 1;
        en = 0;
        w_val = 0;
        x_val = 0;

        #20;
        reset = 0;
        #10;

        // Apply some test inputs
        en = 1;
        w_val = 8'd12; x_val = 8'd34; // 12 * 34 = 408
        #10;
        w_val = 8'd56; x_val = 8'd78; // 56 * 78 = 4368
        #10;
        w_val = 8'd128; x_val = 8'd64; // 128 * 64 = 8192
        #10;
        w_val = 8'd255; x_val = 8'd2; // 255 * 2 = 510
        #10;
        en = 0;
        #30;

        // Expected sum: 408 + 4368 + 8192 + 510 = 13478
        $display("====================================================");
        $display("  RTL Simulation Results");
        $display("====================================================");
        $display("  MAC Accumulated Output: %d (Expected: 13478)", mac_acc);
        $display("  QSM Accumulated Output: %d (Expected: 13478)", qsm_acc);
        if (mac_acc == 13478 && qsm_acc == 13478) begin
            $display("  ✅ SUCCESS: Both units match exactly!");
        end else begin
            $display("  ❌ FAILURE: Output mismatch!");
        end
        $display("====================================================");
        $finish;
    end

endmodule
