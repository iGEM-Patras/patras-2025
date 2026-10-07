library(deSolve)
library(ggplot2)

# 1. Parameters (Units: M, s)
#
# kcat: Feng et al. (2024) report that each active Cas13a cleaved 330-400 reporter
#       molecules in 1 h. The midpoint, 365 per hour, gives 365/3600 = 0.1014 s^-1.
#       Measured on a linear reporter RNA at 23-32 C, not on the structured H0
#       hairpin used here and not at 37 C, so this is an approximation.
# Km:   Feng et al. (2024) quote K_M(Cas13a) ~ 0.2-7 uM from the literature. The
#       midpoint of that range, 3.6-3.7 uM, is used here.
params <- c(
  k1f  = 6.23e+06,
  k1r  = 1.34e-03,
  k2f  = 6.23e+06,
  k2r  = 3.40e+04,
  kcat = 365 / 3600,   # see note below
  Km   = 3.7e-06     # 3.7 uM, see note below
)

# 2. ODE system
cha_system_fixed <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    # Initiator production by Cas13a
    V_cas = (kcat * miRNA * H0) / (Km + H0)

    # CHA reactions
    V1 = k1f * I * H1 - k1r * I_H1
    V2 = k2f * I_H1 * H2 - k2r * H1_H2 * I

    dmiRNA <- 0
    dH0    <- -V_cas
    dI     <- V_cas - V1 + V2
    dH1    <- -V1
    dH2    <- -V2
    dI_H1  <- V1 - V2
    dH1_H2 <- V2

    return(list(c(dmiRNA, dH0, dI, dH1, dH2, dI_H1, dH1_H2)))
  })
}

# 3. Initial concentrations
h_conc      <- 50e-9     # 50 nM each, the final concentration in the CHA reaction
                         # (5 uL of a 1 uM working stock in a 100 uL reaction)
h0_conc     <- 200e-9    # 200 nM, the final concentration of H0 in the CHA reaction.
                         # The protocol pipettes 4 uL of a 5 uM stock into the 20 uL
                         # Cas13a reaction (20 pmol, i.e. 1 uM there); all 20 uL are
                         # then carried into the 100 uL CHA reaction, so the same
                         # 20 pmol sit in 100 uL = 200 nM.
mir_healthy <- 1e-12     # 1 pM
fold_sepsis <- 5.0       # 5.0x
mir_sepsis  <- mir_healthy * fold_sepsis   # 5 pM

init_healthy <- c(miRNA = mir_healthy, H0 = h0_conc, I = 0, H1 = h_conc, H2 = h_conc, I_H1 = 0, H1_H2 = 0)
init_sepsis  <- c(miRNA = mir_sepsis,  H0 = h0_conc, I = 0, H1 = h_conc, H2 = h_conc, I_H1 = 0, H1_H2 = 0)

# 4. Time (1 hour)
times <- seq(0, 3600, by = 1)

# 5. Solve
# NOTE: concentrations are in M (1e-12 ... 1e-6), so the default atol (1e-6) is far too
# large and produces numerical noise / a spike at t = 0. Use tight tolerances.
out_healthy <- as.data.frame(ode(y = init_healthy, times = times, func = cha_system_fixed, parms = params,
                                 method = "lsoda", rtol = 1e-8, atol = 1e-18))
out_sepsis  <- as.data.frame(ode(y = init_sepsis,  times = times, func = cha_system_fixed, parms = params,
                                 method = "lsoda", rtol = 1e-8, atol = 1e-18))

lab_healthy <- "Healthy (~1 pM)"
lab_sepsis  <- "Sepsis (5.0x)"

out_healthy$Condition <- lab_healthy
out_sepsis$Condition  <- lab_sepsis
df <- rbind(out_healthy, out_sepsis)
df$Condition <- factor(df$Condition, levels = c(lab_healthy, lab_sepsis))

# 6. Plot (colorblind-friendly, solid lines)
p <- ggplot(df, aes(x = time / 60, y = H1_H2 * 1e9, color = Condition)) +
  geom_line(linewidth = 1.2, linetype = "solid") +
  scale_color_manual(values = setNames(c("#0072B2", "#D55E00"), c(lab_healthy, lab_sepsis))) +
  labs(
    title = "CHA: H1-H2 duplex formation",
    x = "Time (minutes)",
    y = "H1-H2 duplex (nM)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 12),
    axis.title = element_text(size = 11),
    legend.title = element_blank()
  )

print(p)
ggsave("results/kinetic_models/Fig7_CHA_duplex.png", p, width = 8, height = 5, dpi = 300)
ggsave("results/kinetic_models/Fig7_CHA_duplex.pdf", p, width = 8, height = 5)
# =============================================================================
# 7. Key numbers quoted in the manuscript
# =============================================================================
ceiling_nM <- h_conc * 1e9          # duplex cannot exceed the limiting hairpin

dup <- function(out) out$H1_H2 * 1e9
first_time <- function(out, target) {
  v <- dup(out); i <- which(v >= target)[1]
  if (is.na(i)) NA else out$time[i] / 60
}

cat(sprintf("\nH1 = H2 = %.0f nM  ->  duplex ceiling %.0f nM\n", ceiling_nM, ceiling_nM))
cat(sprintf("%8s %12s %12s %8s\n", "minutes", "healthy nM", "sepsis nM", "ratio"))
for (m in c(10, 20, 30, 45, 60)) {
  i <- which(out_healthy$time == m * 60)
  cat(sprintf("%8d %12.2f %12.2f %8.2f\n", m,
              dup(out_healthy)[i], dup(out_sepsis)[i],
              dup(out_sepsis)[i] / dup(out_healthy)[i]))
}

cat("\nSeparation (absolute difference, septic minus healthy):\n")
d <- dup(out_sepsis) - dup(out_healthy)
for (thr in c(1, 2, 5, 10)) {
  i <- which(d >= thr)[1]
  if (!is.na(i)) cat(sprintf("  >= %2d nM reached at %5.1f min\n", thr, out_sepsis$time[i] / 60))
}
cat(sprintf("  largest difference within the 60 min window: %.1f nM at %.0f min\n", max(d), out_sepsis$time[which.max(d)] / 60))

cat("\nNote: the microRNA input is picomolar, so the Cas13a step is rate-limiting and the\n")
cat("circuit stays far below saturation. The model is therefore close to linear in the\n")
cat("H1/H2 concentration: raising both from 50 nM to 1 uM multiplies every duplex value\n")
cat("by about 20 (20.06 at 60 min) and leaves the septic-to-healthy ratios unchanged.\n")
cat("H0 enters through the Michaelis-Menten term instead, so changing it alters the\n")
cat("ratios as well as the amplitude.\n")
