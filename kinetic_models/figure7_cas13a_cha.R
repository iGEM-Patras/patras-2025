library(deSolve)
library(ggplot2)

# 1. Parameters (Units: M, s)
params <- c(
  k1f  = 6.23e+06,
  k1r  = 1.34e-03,
  k2f  = 6.23e+06,
  k2r  = 3.40e+04,
  kcat = 0.086,
  Km   = 3.0e-06     # 3 uM
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
h_conc      <- 1.0e-06   # 1 uM
h0_conc     <- 5.0e-07   # 500 nM
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
# ggsave("figure7.png", p, width = 8, height = 5, dpi = 300)