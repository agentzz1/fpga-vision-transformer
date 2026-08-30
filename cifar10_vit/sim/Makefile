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

# RTL sources in dependency order (packages first).
RTL_SRC := \
	$(RTL_DIR)/uart_rx.vhd \
	$(RTL_DIR)/uart_tx.vhd \
	$(RTL_DIR)/seg7_display.vhd

TB_SRC := $(wildcard $(TB_DIR)/tb_*.vhd)
TB_TOPS := $(basename $(notdir $(TB_SRC)))

# Simulations stop at this virtual time if a testbench never terminates itself.
STOP_TIME ?= 20ms

.PHONY: all analyse run clean wave list $(TB_TOPS)

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
run: analyse
	@fail=0; \
	for tb in $(TB_TOPS); do \
		echo "=== $$tb ==="; \
		$(GHDL) -e $(GHDLFLAGS) -o $(WORKDIR)/$$tb $$tb || { fail=1; continue; }; \
		( cd $(WORKDIR) && $(GHDL) -r $(STD) --workdir=. $$tb --stop-time=$(STOP_TIME) ) \
			|| { echo "*** $$tb FAILED"; fail=1; }; \
	done; \
	if [ $$fail -ne 0 ]; then echo "SOME TESTBENCHES FAILED"; exit 1; fi; \
	echo "ALL TESTBENCHES PASSED"

# Run a single testbench by name: make tb_uart_loopback
$(TB_TOPS): analyse
	$(GHDL) -e $(GHDLFLAGS) -o $(WORKDIR)/$@ $@
	cd $(WORKDIR) && $(GHDL) -r $(STD) --workdir=. $@ --stop-time=$(STOP_TIME)

# Waveform dump for the testbench named in TB=<name>
wave: analyse
	@test -n "$(TB)" || { echo "usage: make wave TB=tb_uart_loopback"; exit 1; }
	$(GHDL) -e $(GHDLFLAGS) -o $(WORKDIR)/$(TB) $(TB)
	cd $(WORKDIR) && $(GHDL) -r $(STD) --workdir=. $(TB) \
		--wave=$(TB).ghw --stop-time=$(STOP_TIME)
	$(GTKWAVE) $(WORKDIR)/$(TB).ghw &

list:
	@echo "RTL sources:"; for f in $(RTL_SRC); do echo "  $$f"; done
	@echo "testbenches:"; for t in $(TB_TOPS); do echo "  $$t"; done

clean:
	rm -rf $(WORKDIR)
	rm -f *.cf *.o *.ghw *.vcd
