\ Model knapsack_20_1_1
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x81b32103c6e3a266
Minimize
  - 764 y[0] - 650 y[1] - 744 y[2] - 531 y[3] - 302 y[4] - 780 y[5]
   - 983 y[6] - 800 y[7] - 376 y[8] - 1062 y[9] - 646 y[10] - 797 y[11]
   - 489 y[12] - 1094 y[13] - 328 y[14] - 832 y[15] - 145 y[16] - 435 y[17]
   - 950 y[18] - 991 y[19]
Subject To
 primal_upper: 664 y[0] + 550 y[1] + 644 y[2] + 431 y[3] + 202 y[4]
   + 680 y[5] + 883 y[6] + 700 y[7] + 276 y[8] + 962 y[9] + 546 y[10]
   + 697 y[11] + 389 y[12] + 994 y[13] + 228 y[14] + 732 y[15] + 45 y[16]
   + 335 y[17] + 850 y[18] + 891 y[19] + 66 z[0] + 55 z[1] + 64 z[2]
   + 43 z[3] + 20 z[4] + 68 z[5] + 88 z[6] + 70 z[7] + 28 z[8] + 96 z[9]
   + 55 z[10] + 70 z[11] + 39 z[12] + 99 z[13] + 23 z[14] + 73 z[15]
   + 4 z[16] + 34 z[17] + 85 z[18] + 89 z[19] <= 5849
 primal_lower0: - 26 y[0] - 18 y[1] - 2 y[2] - 25 y[3] - 7 y[4] - 23 y[5]
   - 12 y[6] - 25 y[7] - 26 y[8] - 10 y[9] - 4 y[10] - 21 y[11] - 36 y[12]
   - 32 y[13] - 41 y[14] - 21 y[15] - 27 y[16] - 41 y[18] - 27 y[19]
   + 493 u[0] + 574 u[1] + 822 u[2] + 514 u[3] + 936 u[4] + 440 u[5]
   + 69 u[6] + 512 u[7] + 874 u[8] + 481 u[9] + 771 u[10] + 861 u[11]
   + 99 u[12] + 953 u[13] + 181 u[14] + 311 u[15] + 211 u[16] + 862 u[17]
   + 188 u[18] + 782 u[19] <= 953
 mccormick1[0]: - u[0] + z[0] <= 0
 mccormick1[1]: - u[1] + z[1] <= 0
 mccormick1[2]: - u[2] + z[2] <= 0
 mccormick1[3]: - u[3] + z[3] <= 0
 mccormick1[4]: - u[4] + z[4] <= 0
 mccormick1[5]: - u[5] + z[5] <= 0
 mccormick1[6]: - u[6] + z[6] <= 0
 mccormick1[7]: - u[7] + z[7] <= 0
 mccormick1[8]: - u[8] + z[8] <= 0
 mccormick1[9]: - u[9] + z[9] <= 0
 mccormick1[10]: - u[10] + z[10] <= 0
 mccormick1[11]: - u[11] + z[11] <= 0
 mccormick1[12]: - u[12] + z[12] <= 0
 mccormick1[13]: - u[13] + z[13] <= 0
 mccormick1[14]: - u[14] + z[14] <= 0
 mccormick1[15]: - u[15] + z[15] <= 0
 mccormick1[16]: - u[16] + z[16] <= 0
 mccormick1[17]: - u[17] + z[17] <= 0
 mccormick1[18]: - u[18] + z[18] <= 0
 mccormick1[19]: - u[19] + z[19] <= 0
 mccormick2[0]: - y[0] + z[0] <= 0
 mccormick2[1]: - y[1] + z[1] <= 0
 mccormick2[2]: - y[2] + z[2] <= 0
 mccormick2[3]: - y[3] + z[3] <= 0
 mccormick2[4]: - y[4] + z[4] <= 0
 mccormick2[5]: - y[5] + z[5] <= 0
 mccormick2[6]: - y[6] + z[6] <= 0
 mccormick2[7]: - y[7] + z[7] <= 0
 mccormick2[8]: - y[8] + z[8] <= 0
 mccormick2[9]: - y[9] + z[9] <= 0
 mccormick2[10]: - y[10] + z[10] <= 0
 mccormick2[11]: - y[11] + z[11] <= 0
 mccormick2[12]: - y[12] + z[12] <= 0
 mccormick2[13]: - y[13] + z[13] <= 0
 mccormick2[14]: - y[14] + z[14] <= 0
 mccormick2[15]: - y[15] + z[15] <= 0
 mccormick2[16]: - y[16] + z[16] <= 0
 mccormick2[17]: - y[17] + z[17] <= 0
 mccormick2[18]: - y[18] + z[18] <= 0
 mccormick2[19]: - y[19] + z[19] <= 0
 mccormick3[0]: y[0] + u[0] - z[0] <= 1
 mccormick3[1]: y[1] + u[1] - z[1] <= 1
 mccormick3[2]: y[2] + u[2] - z[2] <= 1
 mccormick3[3]: y[3] + u[3] - z[3] <= 1
 mccormick3[4]: y[4] + u[4] - z[4] <= 1
 mccormick3[5]: y[5] + u[5] - z[5] <= 1
 mccormick3[6]: y[6] + u[6] - z[6] <= 1
 mccormick3[7]: y[7] + u[7] - z[7] <= 1
 mccormick3[8]: y[8] + u[8] - z[8] <= 1
 mccormick3[9]: y[9] + u[9] - z[9] <= 1
 mccormick3[10]: y[10] + u[10] - z[10] <= 1
 mccormick3[11]: y[11] + u[11] - z[11] <= 1
 mccormick3[12]: y[12] + u[12] - z[12] <= 1
 mccormick3[13]: y[13] + u[13] - z[13] <= 1
 mccormick3[14]: y[14] + u[14] - z[14] <= 1
 mccormick3[15]: y[15] + u[15] - z[15] <= 1
 mccormick3[16]: y[16] + u[16] - z[16] <= 1
 mccormick3[17]: y[17] + u[17] - z[17] <= 1
 mccormick3[18]: y[18] + u[18] - z[18] <= 1
 mccormick3[19]: y[19] + u[19] - z[19] <= 1
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9] y[10] y[11] y[12] y[13]
 y[14] y[15] y[16] y[17] y[18] y[19] u[0] u[1] u[2] u[3] u[4] u[5] u[6]
 u[7] u[8] u[9] u[10] u[11] u[12] u[13] u[14] u[15] u[16] u[17] u[18] u[19]
 z[0] z[1] z[2] z[3] z[4] z[5] z[6] z[7] z[8] z[9] z[10] z[11] z[12] z[13]
 z[14] z[15] z[16] z[17] z[18] z[19]
End
