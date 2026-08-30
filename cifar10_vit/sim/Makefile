# =============================================================================
# GHDL build/simulation for the CIFAR-10 ViT accelerator (cifar10_vit)
#
#   make            - analyse everything and run every testbench
#   make analyse    - analyse RTL + testbenches only (syntax/elaboration check)
#   make tb_<name>  - build and run one testbench, e.g. make tb_uart_loopback
#   make wave       - re-run the last testbench dumping a GHW waveform
#   make clean      - remove build artefacts
#
# GHDL 4.x with --std=08 (VHDL-2008). Sources live in ../rtl, testbenches here.
# =============================================================================

GHDL     ?= ghdl
GTKWAVE  ?= gtkwave
STD      := --std=08
WORKDIR  := work
GHDLFLAGS := $(STD) --workdir=$(WORKDIR)

RTL_DIR := ../rtl
TB_DIR  := .

# Package first: everything else compiles against it.
RTL_SRC := \
	$(RTL_DIR)/vit_pkg.vhd \
	$(RTL_DIR)/pe_array.vhd \
	$(RTL_DIR)/act_ram.vhd \
	$(RTL_DIR)/weight_rom.vhd \
	$(RTL_DIR)/gemm_seq.vhd \
	$(RTL_DIR)/softmax.vhd \
	$(RTL_DIR)/layernorm.vhd \
	$(RTL_DIR)/elem_op.vhd \
	$(RTL_DIR)/stage_map.vhd \
	$(RTL_DIR)/uart_rx.vhd \
	$(RTL_DIR)/uart_tx.vhd \
	$(RTL_DIR)/seg7_display.vhd

TB_SRC := $(wildcard $(TB_DIR)/tb_*.vhd)
TB_TOPS := $(basename $(notdir $(TB_SRC)))

# Simulations stop at this virtual time if a testbench never terminates itself.
STOP_TIME ?= 20ms

.PHONY: all analyse run clean wave list vectors $(TB_TOPS)

all: run

$(WORKDIR):
	@mkdir -p $(WORKDIR)

# ---------------------------------------------------------------------------
# Analyse: compile RTL then testbenches. Fails loudly on the first error.
# ---------------------------------------------------------------------------
analyse: $(WORKDIR)
	@for f in $(RTL_SRC); do \
		echo "[ghdl -a] $$f"; \
		$(GHDL) -a $(GHDLFLAGS) $$f || exit 1; \
	done
	@for f in $(TB_SRC); do \
		echo "[ghdl -a] $$f"; \
		$(GHDL) -a $(GHDLFLAGS) $$f || exit 1; \
	done
	@echo "analyse OK"

# ---------------------------------------------------------------------------
# Run every testbench and summarise. A testbench reports failure through
# `severity failure`, which makes ghdl -r exit non-zero, so a broken design
# fails the build rather than printing a green line.
# ---------------------------------------------------------------------------
run: vectors analyse
	@fail=0; \
	for tb in $(TB_TOPS); do \
		echo "=== $$tb ==="; \
		$(GHDL) -e $(GHDLFLAGS) -o $(WORKDIR)/$$tb $$tb || { fail=1; continue; }; \
		$(GHDL) -r $(GHDLFLAGS) $$tb --stop-time=$(STOP_TIME) \
			|| { echo "*** $$tb FAILED"; fail=1; }; \
	done; \
	if [ $$fail -ne 0 ]; then echo "SOME TESTBENCHES FAILED"; exit 1; fi; \
	echo "ALL TESTBENCHES PASSED"

# Run a single testbench by name: make tb_uart_loopback
$(TB_TOPS): vectors analyse
	$(GHDL) -e $(GHDLFLAGS) -o $(WORKDIR)/$@ $@
	$(GHDL) -r $(GHDLFLAGS) $@ --stop-time=$(STOP_TIME)

# Waveform dump for the testbench named in TB=<name>
wave: analyse
	@test -n "$(TB)" || { echo "usage: make wave TB=tb_uart_loopback"; exit 1; }
	$(GHDL) -e $(GHDLFLAGS) -o $(WORKDIR)/$(TB) $(TB)
	$(GHDL) -r $(GHDLFLAGS) $(TB) --wave=$(TB).ghw --stop-time=$(STOP_TIME)
	$(GTKWAVE) $(TB).ghw &

list:
	@echo "RTL sources:"; for f in $(RTL_SRC); do echo "  $$f"; done
	@echo "testbenches:"; for t in $(TB_TOPS); do echo "  $$t"; done

# Regenerate the simulation vectors the testbenches read. They are derived
# from the Python golden model rather than checked in, so the reference can
# never go stale relative to the code that defines it.
VECTORS := vectors/gemm_meta.txt vectors/softmax_vectors.txt \
	vectors/layernorm_vectors.txt vectors/elem_vectors.txt

vectors: $(VECTORS)

$(VECTORS):
	cd ../sw && python3 gen_gemm_vectors.py --k 64 --n 64 --shift 7 --scale 24
	cd ../sw && python3 gen_softmax_vectors.py
	cd ../sw && python3 gen_layernorm_vectors.py
	cd ../sw && python3 gen_elem_vectors.py

clean:
	rm -rf $(WORKDIR) work_*
	rm -f *.cf *.o *.ghw *.vcd
