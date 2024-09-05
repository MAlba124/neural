use std::time::Instant;

use neural::matrix::{init, Matrix};

fn main() {
    init();

    let sizes = vec![8, 64, 128, 256, 512, 1024, 2048];

    for n in sizes {
        let mut a = Matrix::from_slice_cm(&vec![1.123; n * n], n, n);
        let mut b = Matrix::from_slice_cm(&vec![1.123; n * n], n, n);
        a.randomize();
        b.randomize();
        let mut c = Matrix::new(n, n);

        let start = Instant::now();
        a.product_into(&b, &mut c);
        let end = start.elapsed();

        let gflop = (2 * n * n * n) as f64 * 1e-9;
        println!(
            "Size: {n}x{n} Perf: {:.0} GFLOP/S",
            gflop / end.as_secs_f64(),
        );
    }
}

// 21:46 30.08.2024:
//   Size: 8x8 Perf: 0.84 GFLOP/S
//   Size: 64x64 Perf: 1.64 GFLOP/S
//   Size: 128x128 Perf: 1.36 GFLOP/S
//   Size: 256x256 Perf: 1.30 GFLOP/S
//   Size: 512x512 Perf: 1.27 GFLOP/S
//   Size: 1024x1024 Perf: 1.21 GFLOP/S
//   Size: 2048x2048 Perf: 0.58 GFLOP/S

// 22:48 30.08.2024:
//   Size: 8x8 Perf: 0.62 GFLOP/S
//   Size: 64x64 Perf: 3.44 GFLOP/S
//   Size: 128x128 Perf: 2.51 GFLOP/S
//   Size: 256x256 Perf: 2.22 GFLOP/S
//   Size: 512x512 Perf: 2.28 GFLOP/S
//   Size: 1024x1024 Perf: 2.30 GFLOP/S
//   Size: 2048x2048 Perf: 2.02 GFLOP/S

//                                                                                    WNXXXXXXXXXXNW
//                                                                                WXXXXKXXXNNNXXXXKXXXXXXNW
//                                                                            WNXKKXNW               WNXXKKXXNW
//                                                                        WNKKXXNW                        WNXXXKKN
//                                                                    WXKXKXXW                                 WK0KW
//                                                                  WXKKN                                        WKOX
//                                                                 WK0N                        WNNXXXKXXN          XOX
//                                                                WKKW               WWNXXXXXXXXXKKKKKXXXXKXNW      K0W
//                                                               WKKW          WNXXXXXXXXXK0KXXXXXXXXK0K00KNN       W0X
//                                                              WKKW     WWNXXXKK00XXXXXXKKXNNNNNXXXXXXXXKKXW        KK
//                                                              XKWWNXXXXKKK00KXXXXXNXKKKXXXXXXXNNNW       W         X0W
//                                                              KX WXKKKXXXK0O0XXXXNXXXXNNNW           WWWWW         X0N
//  19:06 05.09.2024:                                           W0X    WNXKXXXXNNNWW            WXXXXXXXXXXXXXXXW     W0X
//  Size: 8x8 Perf: 0 GFLOP/S                                   N0X      W                      WNNWWW        WW       KK
//  Size: 64x64 Perf: 19 GFLOP/S                                WKX                                                    K0W
//  Size: 128x128 Perf: 210 GFLOP/S                             KK      WWNXXXXXXXXXXW          NKN                   X0N
//  Size: 256x256 Perf: 23 GFLOP/S                              XKW  WXXKKKXNWW  WWNNWW    WW WKO0KOdokKKX            X0X
//  Size: 512x512 Perf: 655 GFLOP/S                             NKX  WXN      WWNXXXXK0N   XX XkOKk'  .x0X            N0X
//  Size: 1024x1024 Perf: 1004 GFLOP/S                          KKW      WXKOc,,c0KOOkK   KKWWX0X0dlox0N             W0K
//  Size: 2048x2048 Perf: 1045 GFLOP/S                          X0X      WO0x.  .kXk0KK   K0W  WWW                    KK
//                                                               WKKW      NKOl;;l0KKWXKW  N0X                         XK
//                                                                N0X                 XKN   K0N                        XKW
//                                     NXNW                        KK                 NKX   W00W                       KK
//                                    Nkk0O0KXXNNW                 X0N                X0N    WKOKW                    WKX
//                                    WKO0KKKK0KK00XW              W0KW              X0XW      N0K                    XON
//                                      WKO0N    WX0KKKKKKNW        XOX              K0        W0X                   NkdK
//                                        XOX         WNXKKKXKKKKKKXXk0              N0KXXKXXXXKKW                 WWkck0X
//                                       W0ON                WNXXKKKKOONWWW           WXXXNWWWWWW               WWNXXOckKOXW
//                                      W0kX                         WXkO0XWW        WNWWNNXXW WNW W          WN00KOxkkoOKKKXXNW
//                                      KkKNNXKNW                     W0kOkON       WKKWKx000N N0XWKX   W    WXkdOX0ddk0KN WNXXXXKXNW
//                                      X0KKKK0000KN                    NKxkN     WKOKWKl,coodocdXWXKXNWKX   NOdxO0xOKO0X W     WNXXX
//                                              WX000KN                   WK0XNN WKKXW0,         ,xXWXXWKOKNNKK0OKdoOKX0X W
//                                                 WX0O0N                  WXOdONOOXW d.           'kW  NkxKXXWNXdlO0XXKN
//                                                    N000XW                 0okXOxxK O'          .:kW   X0K0XXKOdxKW
//                                                      WX000N              WNX0OdlxKNWx.       .:xOX    NNNNXkdk0XW
//                                                         N0OKW               0ddkO0X Nxc::ccclkO0X      WWKoxkdKW
//                                                           N00NW             XXXOxON  NKKKKKKK0XW       Xkll000W
//                                                            WXKKXW              WOoOXW  WNNXNW        W0dll0WWW
//                                                               X0KN              W0dxKNXN         W  NkoOOOXW
//                                                                WX0KW             WXOOKkkK0X WKOXXO0KxoOKXW
//                                                                  NK0KW             NNWN0Oolxko;dklok0XNNW
//                                                                    WK00KN               WXxxKKO0XKXXXN
//                                                                      WNKOKW               WNW
//                                                                         NOKW
//                                                                          N0K
//                                                                           KKW
//                                                                           X0N
//                                                                           W0K
//                                                                            X0N
//                                                                            WK0N
//                                                                             W00N
//                                                                              W0KW
//                                                                               W0K
//                                                                                K0W
