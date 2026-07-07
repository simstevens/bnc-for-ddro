\ Model knapsack_40_1_2
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xe2fdb6ddddddb322
Minimize
  10 x[0] + 10 x[1] + 10 x[2] + 10 x[3] + 10 x[4] + 10 x[5] + 10 x[6]
   + 10 x[7] + 10 x[8] + 10 x[9] + 10 x[10] + 10 x[11] + 10 x[12]
   + 10 x[13] + 10 x[14] + 10 x[15] + 10 x[16] + 10 x[17] + 10 x[18]
   + 10 x[19] + 10 x[20] + 10 x[21] + 10 x[22] + 10 x[23] + 10 x[24]
   + 10 x[25] + 10 x[26] + 10 x[27] + 10 x[28] + 10 x[29] + 10 x[30]
   + 10 x[31] + 10 x[32] + 10 x[33] + 10 x[34] + 10 x[35] + 10 x[36]
   + 10 x[37] + 10 x[38] + 10 x[39] - 991 y[0] - 776 y[1] - 329 y[2]
   - 565 y[3] - 819 y[4] - 186 y[5] - 707 y[6] - 413 y[7] - 840 y[8]
   - 556 y[9] - 216 y[10] - 420 y[11] - 404 y[12] - 51 y[13] - 320 y[14]
   - 653 y[15] - 551 y[16] - 947 y[17] - 20 y[18] - 139 y[19] - 891 y[20]
   - 144 y[21] - 837 y[22] - 865 y[23] - 562 y[24] - 936 y[25] - 529 y[26]
   - 993 y[27] - 35 y[28] - 402 y[29] - 710 y[30] - 505 y[31] - 533 y[32]
   - 656 y[33] - 917 y[34] - 158 y[35] - 506 y[36] - 384 y[37] - 624 y[38]
   - 237 y[39]
Subject To
 primal_upper: 717 y[0] + 313 y[1] + 959 y[2] + 176 y[3] + 129 y[4]
   + 911 y[5] + 434 y[6] + 73 y[7] + 474 y[8] + 263 y[9] + 785 y[10]
   + 64 y[11] + 329 y[12] + 113 y[13] + 714 y[14] + 754 y[15] + 922 y[16]
   + 109 y[17] + 6 y[18] + 427 y[19] + 194 y[20] + 981 y[21] + 297 y[22]
   + 972 y[23] + 218 y[24] + 445 y[25] + 471 y[26] + 653 y[27] + 699 y[28]
   + 204 y[29] + 319 y[30] + 121 y[31] + 545 y[32] + 530 y[33] + 713 y[34]
   + 518 y[35] + 265 y[36] + 42 y[37] + 9 y[38] + 420 y[39] + 72 z[0]
   + 31 z[1] + 96 z[2] + 18 z[3] + 13 z[4] + 91 z[5] + 43 z[6] + 7 z[7]
   + 47 z[8] + 26 z[9] + 78 z[10] + 6 z[11] + 33 z[12] + 11 z[13]
   + 71 z[14] + 75 z[15] + 92 z[16] + 11 z[17] + z[18] + 43 z[19]
   + 19 z[20] + 98 z[21] + 30 z[22] + 97 z[23] + 22 z[24] + 44 z[25]
   + 47 z[26] + 65 z[27] + 70 z[28] + 20 z[29] + 32 z[30] + 12 z[31]
   + 54 z[32] + 53 z[33] + 71 z[34] + 52 z[35] + 26 z[36] + 4 z[37] + z[38]
   + 42 z[39] <= 5762
 primal_lower0: 683 u[0] + 172 u[1] + 830 u[2] + 779 u[3] + 360 u[4]
   + 553 u[5] + 865 u[6] + 611 u[7] + 817 u[8] + 450 u[9] + 983 u[10]
   + 715 u[11] + 576 u[12] + 972 u[13] + 440 u[14] + 876 u[15] + 601 u[16]
   + 616 u[17] + 45 u[18] + 902 u[19] + 145 u[20] + 15 u[21] + 376 u[22]
   + 30 u[23] + 424 u[24] + 96 u[25] + 870 u[26] + 911 u[27] + 873 u[28]
   + 519 u[29] + 749 u[30] + 11 u[31] + 655 u[32] + 861 u[33] + 656 u[34]
   + 145 u[35] + 312 u[36] + 396 u[37] + 552 u[38] + 958 u[39] <= 983
 interdiction[0]: x[0] + u[0] <= 1
 interdiction[1]: x[1] + u[1] <= 1
 interdiction[2]: x[2] + u[2] <= 1
 interdiction[3]: x[3] + u[3] <= 1
 interdiction[4]: x[4] + u[4] <= 1
 interdiction[5]: x[5] + u[5] <= 1
 interdiction[6]: x[6] + u[6] <= 1
 interdiction[7]: x[7] + u[7] <= 1
 interdiction[8]: x[8] + u[8] <= 1
 interdiction[9]: x[9] + u[9] <= 1
 interdiction[10]: x[10] + u[10] <= 1
 interdiction[11]: x[11] + u[11] <= 1
 interdiction[12]: x[12] + u[12] <= 1
 interdiction[13]: x[13] + u[13] <= 1
 interdiction[14]: x[14] + u[14] <= 1
 interdiction[15]: x[15] + u[15] <= 1
 interdiction[16]: x[16] + u[16] <= 1
 interdiction[17]: x[17] + u[17] <= 1
 interdiction[18]: x[18] + u[18] <= 1
 interdiction[19]: x[19] + u[19] <= 1
 interdiction[20]: x[20] + u[20] <= 1
 interdiction[21]: x[21] + u[21] <= 1
 interdiction[22]: x[22] + u[22] <= 1
 interdiction[23]: x[23] + u[23] <= 1
 interdiction[24]: x[24] + u[24] <= 1
 interdiction[25]: x[25] + u[25] <= 1
 interdiction[26]: x[26] + u[26] <= 1
 interdiction[27]: x[27] + u[27] <= 1
 interdiction[28]: x[28] + u[28] <= 1
 interdiction[29]: x[29] + u[29] <= 1
 interdiction[30]: x[30] + u[30] <= 1
 interdiction[31]: x[31] + u[31] <= 1
 interdiction[32]: x[32] + u[32] <= 1
 interdiction[33]: x[33] + u[33] <= 1
 interdiction[34]: x[34] + u[34] <= 1
 interdiction[35]: x[35] + u[35] <= 1
 interdiction[36]: x[36] + u[36] <= 1
 interdiction[37]: x[37] + u[37] <= 1
 interdiction[38]: x[38] + u[38] <= 1
 interdiction[39]: x[39] + u[39] <= 1
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
 mccormick1[20]: - u[20] + z[20] <= 0
 mccormick1[21]: - u[21] + z[21] <= 0
 mccormick1[22]: - u[22] + z[22] <= 0
 mccormick1[23]: - u[23] + z[23] <= 0
 mccormick1[24]: - u[24] + z[24] <= 0
 mccormick1[25]: - u[25] + z[25] <= 0
 mccormick1[26]: - u[26] + z[26] <= 0
 mccormick1[27]: - u[27] + z[27] <= 0
 mccormick1[28]: - u[28] + z[28] <= 0
 mccormick1[29]: - u[29] + z[29] <= 0
 mccormick1[30]: - u[30] + z[30] <= 0
 mccormick1[31]: - u[31] + z[31] <= 0
 mccormick1[32]: - u[32] + z[32] <= 0
 mccormick1[33]: - u[33] + z[33] <= 0
 mccormick1[34]: - u[34] + z[34] <= 0
 mccormick1[35]: - u[35] + z[35] <= 0
 mccormick1[36]: - u[36] + z[36] <= 0
 mccormick1[37]: - u[37] + z[37] <= 0
 mccormick1[38]: - u[38] + z[38] <= 0
 mccormick1[39]: - u[39] + z[39] <= 0
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
 mccormick2[20]: - y[20] + z[20] <= 0
 mccormick2[21]: - y[21] + z[21] <= 0
 mccormick2[22]: - y[22] + z[22] <= 0
 mccormick2[23]: - y[23] + z[23] <= 0
 mccormick2[24]: - y[24] + z[24] <= 0
 mccormick2[25]: - y[25] + z[25] <= 0
 mccormick2[26]: - y[26] + z[26] <= 0
 mccormick2[27]: - y[27] + z[27] <= 0
 mccormick2[28]: - y[28] + z[28] <= 0
 mccormick2[29]: - y[29] + z[29] <= 0
 mccormick2[30]: - y[30] + z[30] <= 0
 mccormick2[31]: - y[31] + z[31] <= 0
 mccormick2[32]: - y[32] + z[32] <= 0
 mccormick2[33]: - y[33] + z[33] <= 0
 mccormick2[34]: - y[34] + z[34] <= 0
 mccormick2[35]: - y[35] + z[35] <= 0
 mccormick2[36]: - y[36] + z[36] <= 0
 mccormick2[37]: - y[37] + z[37] <= 0
 mccormick2[38]: - y[38] + z[38] <= 0
 mccormick2[39]: - y[39] + z[39] <= 0
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
 mccormick3[20]: y[20] + u[20] - z[20] <= 1
 mccormick3[21]: y[21] + u[21] - z[21] <= 1
 mccormick3[22]: y[22] + u[22] - z[22] <= 1
 mccormick3[23]: y[23] + u[23] - z[23] <= 1
 mccormick3[24]: y[24] + u[24] - z[24] <= 1
 mccormick3[25]: y[25] + u[25] - z[25] <= 1
 mccormick3[26]: y[26] + u[26] - z[26] <= 1
 mccormick3[27]: y[27] + u[27] - z[27] <= 1
 mccormick3[28]: y[28] + u[28] - z[28] <= 1
 mccormick3[29]: y[29] + u[29] - z[29] <= 1
 mccormick3[30]: y[30] + u[30] - z[30] <= 1
 mccormick3[31]: y[31] + u[31] - z[31] <= 1
 mccormick3[32]: y[32] + u[32] - z[32] <= 1
 mccormick3[33]: y[33] + u[33] - z[33] <= 1
 mccormick3[34]: y[34] + u[34] - z[34] <= 1
 mccormick3[35]: y[35] + u[35] - z[35] <= 1
 mccormick3[36]: y[36] + u[36] - z[36] <= 1
 mccormick3[37]: y[37] + u[37] - z[37] <= 1
 mccormick3[38]: y[38] + u[38] - z[38] <= 1
 mccormick3[39]: y[39] + u[39] - z[39] <= 1
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34] x[35] x[36] x[37]
 x[38] x[39] y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9] y[10] y[11]
 y[12] y[13] y[14] y[15] y[16] y[17] y[18] y[19] y[20] y[21] y[22] y[23]
 y[24] y[25] y[26] y[27] y[28] y[29] y[30] y[31] y[32] y[33] y[34] y[35]
 y[36] y[37] y[38] y[39] u[0] u[1] u[2] u[3] u[4] u[5] u[6] u[7] u[8] u[9]
 u[10] u[11] u[12] u[13] u[14] u[15] u[16] u[17] u[18] u[19] u[20] u[21]
 u[22] u[23] u[24] u[25] u[26] u[27] u[28] u[29] u[30] u[31] u[32] u[33]
 u[34] u[35] u[36] u[37] u[38] u[39] z[0] z[1] z[2] z[3] z[4] z[5] z[6]
 z[7] z[8] z[9] z[10] z[11] z[12] z[13] z[14] z[15] z[16] z[17] z[18] z[19]
 z[20] z[21] z[22] z[23] z[24] z[25] z[26] z[27] z[28] z[29] z[30] z[31]
 z[32] z[33] z[34] z[35] z[36] z[37] z[38] z[39]
End
