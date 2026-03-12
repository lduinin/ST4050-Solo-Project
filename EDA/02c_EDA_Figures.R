# ==============================================================================
# 02c_EDA_Figures.R — Key figures for write-up (Chapter 3)
# ==============================================================================
# REQUIRES: prepared_data.RData from 01_Data_Preparation.R
# OUTPUT: EDA figures as PDFs
# ==============================================================================

source("00_utils.R")
load("prepared_data.RData")

# ==============================================================================
# FIGURE 1: Claim frequency by Driver Age
# ==============================================================================
# Shows the U-shaped relationship — young and very old drivers have higher
# frequency. This motivates using splines rather than a linear term.

freq_by_drivage <- aggregate(
  cbind(ClaimNb, Exposure) ~ DrivAge, data = learn, FUN = sum
)
freq_by_drivage$Frequency <- freq_by_drivage$ClaimNb / freq_by_drivage$Exposure

pdf("fig_freq_by_drivage.pdf", width = 8, height = 5)
plot(freq_by_drivage$DrivAge, freq_by_drivage$Frequency,
     type = "l", lwd = 2, col = "steelblue",
     xlab = "Driver Age", ylab = "Observed Claim Frequency",
     main = "Claim Frequency by Driver Age")
abline(h = mean(learn$ClaimNb / learn$Exposure), lty = 2, col = "grey40")
dev.off()

# ==============================================================================
# FIGURE 2: Claim frequency by BonusMalus
# ==============================================================================
# Shows strong nonlinear increase. Also visible: most mass is at 50.

freq_by_bm <- aggregate(
  cbind(ClaimNb, Exposure) ~ BonusMalus, data = learn, FUN = sum
)
freq_by_bm$Frequency <- freq_by_bm$ClaimNb / freq_by_bm$Exposure

pdf("fig_freq_by_bonusmalus.pdf", width = 8, height = 5)
par(mar = c(5, 4, 4, 4))
# Bar chart of exposure (shows concentration at 50)
barplot(freq_by_bm$Exposure / 1000, names.arg = freq_by_bm$BonusMalus,
        col = "grey85", border = NA, ylab = "Exposure (thousands)",
        xlab = "Bonus-Malus Level", main = "Claim Frequency and Exposure by Bonus-Malus")
par(new = TRUE)
# Overlay frequency line
plot(seq_along(freq_by_bm$BonusMalus), freq_by_bm$Frequency,
     type = "l", lwd = 2, col = "red", axes = FALSE, xlab = "", ylab = "")
axis(4, col = "red", col.axis = "red")
mtext("Claim Frequency", side = 4, line = 2.5, col = "red")
legend("topleft", legend = c("Exposure", "Frequency"),
       fill = c("grey85", NA), border = c("grey85", NA),
       lty = c(NA, 1), lwd = c(NA, 2), col = c("grey85", "red"),
       bg = "white", cex = 0.9)
dev.off()

# ==============================================================================
# FIGURE 3: Claim frequency by Vehicle Age
# ==============================================================================
# Shows the "new car" spike at VehAge = 0, motivating VehAge categorisation
# and the VehAge:VehBrand interaction (rental/fleet cars).

freq_by_vehage <- aggregate(
  cbind(ClaimNb, Exposure) ~ VehAge, data = learn, FUN = sum
)
freq_by_vehage$Frequency <- freq_by_vehage$ClaimNb / freq_by_vehage$Exposure

pdf("fig_freq_by_vehage.pdf", width = 8, height = 5)
plot(freq_by_vehage$VehAge, freq_by_vehage$Frequency,
     type = "l", lwd = 2, col = "darkgreen",
     xlab = "Vehicle Age (years)", ylab = "Observed Claim Frequency",
     main = "Claim Frequency by Vehicle Age",
     xlim = c(0, 40))
abline(h = mean(learn$ClaimNb / learn$Exposure), lty = 2, col = "grey40")
# Highlight the new car spike
points(0, freq_by_vehage$Frequency[freq_by_vehage$VehAge == 0],
       pch = 19, col = "red", cex = 1.5)
text(2, freq_by_vehage$Frequency[freq_by_vehage$VehAge == 0],
     "New cars (age 0)", pos = 4, col = "red", cex = 0.9)
dev.off()

# ==============================================================================
# FIGURE 4: Distribution of BonusMalus
# ==============================================================================
# Shows extreme concentration at 50 — explains why spline knot placement
# is difficult for this variable.

pdf("fig_dist_bonusmalus.pdf", width = 8, height = 5)
hist(learn$BonusMalus[learn$BonusMalus <= 200], breaks = 100,
     col = "steelblue", border = "white",
     xlab = "Bonus-Malus Level", ylab = "Number of Policies",
     main = "Distribution of Bonus-Malus Levels")
abline(v = 50, lty = 2, col = "red", lwd = 2)
text(55, par("usr")[4] * 0.9, "Base level (50)", col = "red", pos = 4)
dev.off()
