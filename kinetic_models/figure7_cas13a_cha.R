library(deSolve)
library(ggplot2)

# =============================================================================
# Figure 7 - coupled Cas13a / CHA model, run as the two-stage protocol
#
# The experiment is two sequential reactions, not one pot:
#   Stage 1   Cas13a + crRNA + H0 + target in 20 uL, 1 h at 37 C.
#             H1 and H2 are not present, so no CHA can occur.
#   Transfer  all 20 uL are carried into a 100 uL reaction, a 5-fold dilution,
#             and H1 and H2 are added at 50 nM each.
#   Stage 2   CHA proceeds for 3 h at 37 C. Cas13a, crRNA, target and the
#             remaining H0 are all carried over, so initiator production
#             continues alongside the CHA cascade.
#
# Both stages use the same ODE system; during stage 1 the hairpins are simply
# absent (H1 = H2 = 0), which switches the CHA terms off exactly.
#
# Forward rate constants (see Supplementary Table S2):
#   k1f  Hertel et al. (2022) nucleation model applied to the 9-nt H1 toehold,
#        with the scale calibrated so the rate falls within the experimentally
#        observed range for DNA hybridisation. It is a calibrated value, not an
#        independent prediction of the model.
#   k2f  Zhang & Winfree (2009): for toeholds longer than ~6 nt the forward rate
#        reaches a plateau; the H2 toehold is 7 nt, so the plateau value applies.
#   The Hertel model was NOT used for the second reaction.
# =============================================================================

params <- c(
  k1f  = 3.00e+06,     # M^-1 s^-1, Hertel model, H1 toehold (L = 9), calibrated
  k1r  = 1.34e-03,     # s^-1,      NUPACK dG -> Keq
  k2f  = 3.00e+06,     # M^-1 s^-1, Zhang & Winfree plateau, H2 toehold (7 nt)
  k2r  = 3.40e+04,     # M^-1 s^-1, NUPACK dG -> Keq (bimolecular both ways)
  kcat = 365 / 3600,   # s^-1,      Feng et al. (2024), 330-400 reporters per h
  Km   = 3.7e-06       # M,         midpoint of K_M(Cas13a) ~ 0.2-7 uM
)

cha_system <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    V_cas <- (kcat * miRNA * H0) / (Km + H0)      # Cas13a cleaves H0 -> initiator
    V1    <- k1f * I * H1    - k1r * I_H1         # initiator + H1
    V2    <- k2f * I_H1 * H2 - k2r * H1_H2 * I    # I:H1 + H2 -> H1:H2 + initiator

    list(c(dmiRNA  = 0,
           dH0     = -V_cas,
           dI      = V_cas - V1 + V2,
           dH1     = -V1,
           dH2     = -V2,
           dI_H1   = V1 - V2,
           dH1_H2  = V2))
  })
}

# ----------------------------------------------------------------- conditions
T1       <- 60 * 60        # stage 1, 1 hour
T2       <- 180 * 60       # stage 2, 3 hours
DILUTION <- 5              # 20 uL carried into 100 uL
h_conc   <- 50e-9          # H1 and H2, final concentration in the CHA reaction
h0_stage1<- 1.0e-06        # 4 uL of a 5 uM stock in the 20 uL Cas13a reaction
sample_dilution <- 10      # 2 uL of sample in the 20 uL Cas13a reaction

serum_healthy <- 1e-12     # 1 pM in serum
fold_sepsis   <- 5.0
serum_sepsis  <- serum_healthy * fold_sepsis

run_protocol <- function(serum) {
  # --- stage 1: Cas13a only, no hairpins present
  y0 <- c(miRNA = serum / sample_dilution, H0 = h0_stage1,
          I = 0, H1 = 0, H2 = 0, I_H1 = 0, H1_H2 = 0)
  s1 <- as.data.frame(ode(y = y0, times = seq(0, T1, by = 1), func = cha_system,
                          parms = params, method = "lsoda",
                          rtol = 1e-8, atol = 1e-24))

  # --- transfer: 5-fold dilution of everything, then the hairpins are added
  last <- s1[nrow(s1), ]
  y1 <- c(miRNA = last$miRNA / DILUTION, H0 = last$H0 / DILUTION,
          I = last$I / DILUTION, H1 = h_conc, H2 = h_conc,
          I_H1 = 0, H1_H2 = 0)
  s2 <- as.data.frame(ode(y = y1, times = seq(0, T2, by = 1), func = cha_system,
                          parms = params, method = "lsoda",
                          rtol = 1e-8, atol = 1e-24))

  s1$stage <- "1"; s2$stage <- "2"
  s2$time  <- s2$time + T1                     # continuous protocol clock
  rbind(s1, s2)
}

out_h <- run_protocol(serum_healthy); out_h$Condition <- "Healthy (~1 pM)"
out_s <- run_protocol(serum_sepsis);  out_s$Condition <- "Sepsis (5.0x)"
df <- rbind(out_h, out_s)
df$Condition <- factor(df$Condition, levels = c("Healthy (~1 pM)", "Sepsis (5.0x)"))

# ----------------------------------------------------------------------- plot
p <- ggplot(df, aes(x = time / 60, y = H1_H2 * 1e9, color = Condition)) +
  geom_vline(xintercept = T1 / 60, linetype = "dashed",
             colour = "grey55", linewidth = 0.5) +
  annotate("text", x = T1 / 60 + 3, y = Inf, vjust = 1.6, hjust = 0,
           label = "transfer to CHA reaction", size = 3, colour = "grey35") +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Healthy (~1 pM)" = "#0072B2",
                                "Sepsis (5.0x)"   = "#D55E00")) +
  scale_x_continuous(breaks = seq(0, (T1 + T2) / 60, by = 30)) +
  labs(title = "CHA: H1-H2 duplex formation over the two-stage protocol",
       x = "Time (minutes)", y = "H1-H2 duplex (nM)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 12),
        axis.title = element_text(size = 11),
        legend.title = element_blank())

print(p)
ggsave("results/kinetic_models/Fig7_CHA_duplex.png", p, width = 8, height = 5, dpi = 300)
ggsave("results/kinetic_models/Fig7_CHA_duplex.pdf", p, width = 8, height = 5)

# ------------------------------------------------- numbers quoted in the text
dup <- function(d) d$H1_H2 * 1e9
at  <- function(d, mins) dup(d)[which.min(abs(d$time - mins * 60))]

cat(sprintf("\nStage 1: %d min, H0 %.0f nM, no hairpins. Stage 2: %d min, H1 = H2 = %.0f nM.\n",
            T1 / 60, h0_stage1 * 1e9, T2 / 60, h_conc * 1e9))
cat(sprintf("Initiator at the end of stage 1: healthy %.2f pM, septic %.2f pM\n",
            out_h$I[out_h$time == T1][1] * 1e12, out_s$I[out_s$time == T1][1] * 1e12))

cat(sprintf("\n%10s %12s %12s %8s\n", "minutes", "healthy nM", "sepsis nM", "ratio"))
for (m in c(60, 90, 120, 150, 180, 210, 240)) {
  h <- at(out_h, m); s <- at(out_s, m)
  cat(sprintf("%10d %12.3f %12.3f %8.2f\n", m, h, s, ifelse(h > 0, s / h, NA)))
}

st2   <- df$time >= T1
times <- sort(unique(df$time[st2]))
hh    <- sapply(times, function(tt) dup(out_h)[which.min(abs(out_h$time - tt))])
ss    <- sapply(times, function(tt) dup(out_s)[which.min(abs(out_s$time - tt))])
d     <- ss - hh
cat("\nSeparation (septic minus healthy), measured on the protocol clock:\n")
for (thr in c(1, 2, 5, 10)) {
  i <- which(d >= thr)[1]
  if (!is.na(i)) cat(sprintf("  >= %2d nM at %6.1f min (%.1f min into stage 2)\n",
                             thr, times[i] / 60, (times[i] - T1) / 60))
}
r <- ss[hh > 1e-6] / hh[hh > 1e-6]
cat(sprintf("\nSeptic-to-healthy ratio over stage 2: %.2f to %.2f\n", min(r), max(r)))
