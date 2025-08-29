
from __future__ import annotations
import numpy as np
from manim import *

"""
D4 Nested Lattice — Manim Animation (2D Projection)
--------------------------------------------------

We visualize the 4D root lattice D4 via a fixed 2D orthonormal projection,
and illustrate the nested pair D4 ⊂ 2·D4, cosets modulo the coarse lattice,
quantization (Babai rounding), and modulo operation.

Projection used (u1,u2):
  u1 = (1,-1,0,0)/√2
  u2 = (0,0,1,-1)/√2
Coordinates shown = [⟨u1,x⟩, ⟨u2,x⟩] for x ∈ R⁴.

Generator for D4 (columns are basis vectors):
  b1 = (1,-1, 0, 0)
  b2 = (0, 1,-1, 0)
  b3 = (0, 0, 1,-1)
  b4 = (0, 0, 1, 1)

Any integer combination B @ k (k ∈ Z⁴) yields sum(x_i) even, hence x ∈ D4.
Coarse lattice is 2·D4 with basis 2B.
Index [2·D4 : D4] = 2^4 = 16.

Scenes:
  1) D4TitleCard
  2) D4Lattice2DProjection
  3) NestedD4Vs2D4
  4) D4CosetsMod2
  5) D4QuantizationDemo
  6) Mod2D4Demo
"""

# -----------------------------
# Lattice & Projection Helpers
# -----------------------------

# D4 basis (columns)
B_D4 = np.stack([
    np.array([1, -1,  0,  0], dtype=float),
    np.array([0,  1, -1,  0], dtype=float),
    np.array([0,  0,  1, -1], dtype=float),
    np.array([0,  0,  1,  1], dtype=float),
], axis=1)

B_COARSE = 2.0 * B_D4  # 2·D4

# 2D orthonormal projection rows (2x4)
u1 = np.array([1, -1, 0, 0], dtype=float)
u2 = np.array([0,  0, 1, -1], dtype=float)
u1 = u1 / np.linalg.norm(u1)
u2 = u2 / np.linalg.norm(u2)
P = np.stack([u1, u2], axis=0)  # shape (2,4)

def project2(x4: np.ndarray) -> np.ndarray:
    """Project R^4 vector (len 4 or 4D embedded in len 3) to R^2 for plotting."""
    x = x4[:4]
    y = P @ x
    return np.array([y[0], y[1], 0.0])

def gen_lattice_points(B: np.ndarray, kmax: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate lattice points {B k : k ∈ Z^4, -kmax ≤ k_i ≤ kmax}, and their 2D projections.
    Returns (pts4, pts2) lists.
    """
    pts4, pts2 = [], []
    rng = range(-kmax, kmax + 1)
    for a in rng:
        for b in rng:
            for c in rng:
                for d in rng:
                    k = np.array([a, b, c, d], dtype=float)
                    x4 = B @ k
                    pts4.append(x4)
                    pts2.append(project2(x4))
    return pts4, pts2

def draw_points(scene: Scene, pts2: list[np.ndarray], color=WHITE, radius=0.035, opacity=1.0, z_index=1) -> VGroup:
    group = VGroup(*[Dot(p, radius=radius, color=color, fill_opacity=opacity, z_index=z_index) for p in pts2])
    scene.add(group)
    return group

def nearest_lattice_point(B: np.ndarray, x4: np.ndarray) -> np.ndarray:
    """
    Babai rounding in coefficient space (exact for this integer basis).
    Returns lambda = B * round(B^{-1} x).
    """
    coeffs = np.linalg.solve(B, x4)
    rounded = np.round(coeffs)
    v = B @ rounded
    return v

def coset_index_mod_2(B: np.ndarray, x4: np.ndarray) -> int:
    """
    Coset index of x ∈ D4 modulo 2·D4 via coefficient parity.
    If x = B*k with integer k, then k mod 2 ∈ {0,1}^4 → index in {0..15}.
    """
    k = np.linalg.solve(B, x4)
    k_round = np.rint(k).astype(int)
    r = k_round % 2
    return int(r[0] + 2*r[1] + 4*r[2] + 8*r[3])

def make_plane(x_range=(-4, 4, 1), y_range=(-4, 4, 1)) -> NumberPlane:
    plane = NumberPlane(
        x_range=x_range,
        y_range=y_range,
        background_line_style={"stroke_opacity": 0.15},
        axis_config={"stroke_opacity": 0.6},
    )
    return plane

# 16 distinct colors for cosets (repeat Manim palette safely)
COSET16 = [
    YELLOW, RED, TEAL, PURPLE, ORANGE, GREEN, BLUE, PINK,
    MAROON, GOLD, LIGHT_PINK, LIGHT_BROWN, LIGHT_GREY, SEA_GREEN, VIOLET, GRAY_C
]

# -----------------------------
# Scenes
# -----------------------------

class D4TitleCard(Scene):
    def construct(self):
        title = Text("Nested Lattices in D⁴", font_size=72).to_edge(UP)
        subtitle = Text("D⁴ ⊂ 2·D⁴ (2D projection)", font_size=36).next_to(title, DOWN)
        bullets = VGroup(
            Text("D⁴ = {x ∈ ℤ⁴ : sum(x) even}", font_size=28),
            Text("Basis columns B generate all of D⁴", font_size=28),
            Text("Nested pair: fine D⁴ inside coarse 2·D⁴", font_size=28),
            Text("Index [2·D⁴ : D⁴] = 16", font_size=28),
            Text("Quantization & modulo with respect to 2·D⁴", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT).shift(DOWN*0.5)

        eqs = VGroup(
            MathTex(r"D^4=\{x\in\mathbb{Z}^4:\ \sum_i x_i\equiv 0\ (\text{mod }2)\}").scale(0.9),
            MathTex(r"x=B\,k,\ k\in\mathbb{Z}^4,\ \ B=\begin{bmatrix} 1&0&0&0\\ -1&1&0&0\\ 0&-1&1&1\\ 0&0&-1&1 \end{bmatrix}\ \text{(columns are }b_i\text{)}").scale(0.7),
            MathTex(r"Q_{D^4}(y)=B\,\big\lfloor B^{-1}y\rceil,\quad y\bmod 2D^4 = y - Q_{2D^4}(y)").scale(0.9),
            MathTex(r"[\,2D^4:D^4\,]=2^4=16").scale(0.9),
        ).arrange(DOWN, buff=0.32).to_edge(RIGHT).shift(DOWN*0.3)

        self.play(Write(title), FadeIn(subtitle))
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT*0.3) for b in bullets], lag_ratio=0.12))
        self.play(LaggedStart(*[Write(e) for e in eqs], lag_ratio=0.15))
        self.wait(2)


class D4Lattice2DProjection(Scene):
    def construct(self):
        plane = make_plane()
        self.add(plane)

        title = Text("D⁴ lattice (2D projection)", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN*0.5))

        # Generate points
        pts4, pts2 = gen_lattice_points(B_D4, kmax=2)
        draw_points(self, pts2, color=WHITE, radius=0.04, opacity=0.95, z_index=1)

        # Show projection directions
        # Draw arrows for the 2 projection axes using images of 4D basis vectors e1 and e3-e4
        legend = VGroup(
            MathTex(r"u_1=\tfrac{1}{\sqrt{2}}(1,-1,0,0)").scale(0.8),
            MathTex(r"u_2=\tfrac{1}{\sqrt{2}}(0,0,1,-1)").scale(0.8),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT)
        self.play(Write(legend))
        self.wait(2)


class NestedD4Vs2D4(Scene):
    def construct(self):
        plane = make_plane()
        self.add(plane)

        title = Text("Nested pair: D⁴ ⊂ 2·D⁴", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN*0.5))

        # Fine lattice points (D4) in gray
        _, fine2 = gen_lattice_points(B_D4, kmax=2)
        fine = draw_points(self, fine2, color=GRAY_C, radius=0.035, opacity=0.9, z_index=1)

        # Coarse lattice points (2·D4) highlighted
        _, coarse2 = gen_lattice_points(B_COARSE, kmax=2)
        coarse = VGroup(*[Dot(p, radius=0.06, color=YELLOW, fill_opacity=1.0, z_index=2) for p in coarse2])
        self.play(FadeIn(coarse, scale=0.9))

        idx = MathTex(r"[\,2D^4:D^4\,]=16").scale(0.9).to_edge(RIGHT).shift(UP*0.5)
        self.play(Write(idx))
        self.wait(2)


class D4CosetsMod2(Scene):
    def construct(self):
        plane = make_plane()
        self.add(plane)

        title = Text("Cosets of D⁴ / 2·D⁴ (16 classes via coefficient parity)", font_size=32).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN*0.5))

        kmax = 2
        pts = []
        dots_by_coset = [VGroup() for _ in range(16)]
        for a in range(-kmax, kmax+1):
            for b in range(-kmax, kmax+1):
                for c in range(-kmax, kmax+1):
                    for d in range(-kmax, kmax+1):
                        k = np.array([a,b,c,d], dtype=float)
                        x4 = B_D4 @ k
                        p = project2(x4)
                        cid = int((a % 2) + 2*(b % 2) + 4*(c % 2) + 8*(d % 2))
                        dots_by_coset[cid].add(Dot(p, radius=0.06, color=COSET16[cid], fill_opacity=0.95))

        for cid in range(16):
            lbl = MathTex(r"s=" + str(cid)).scale(0.6).to_edge(RIGHT).shift(DOWN*(2.8 - 0.35*cid))
            self.play(FadeIn(dots_by_coset[cid], scale=0.95), Write(lbl))
            self.wait(0.15)

        self.wait(1.2)


class D4QuantizationDemo(Scene):
    def construct(self):
        plane = make_plane()
        self.add(plane)

        title = Text("Quantization in D⁴ vs 2·D⁴ (projection shown)", font_size=32).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN*0.5))

        # Draw coarse lattice for reference
        _, coarse2 = gen_lattice_points(B_COARSE, kmax=2)
        coarse = VGroup(*[Dot(p, radius=0.055, color=YELLOW, fill_opacity=1.0, z_index=2) for p in coarse2])
        self.add(coarse)

        rng = np.random.default_rng(7)
        for _ in range(5):
            # pick random 4D point in a modest range
            y4 = rng.uniform(-2.2, 2.2, size=4)
            y2 = project2(y4)

            q_f = nearest_lattice_point(B_D4, y4)
            q_c = nearest_lattice_point(B_COARSE, y4)
            qf2, qc2 = project2(q_f), project2(q_c)

            y_dot  = Dot(y2,  color=WHITE).set_z_index(3)
            qf_dot = Dot(qf2, color=BLUE).set_z_index(3)
            qc_dot = Dot(qc2, color=RED).set_z_index(3)

            err_f = Arrow(start=y2, end=qf2, buff=0.0, stroke_width=4, color=BLUE_E)
            err_c = Arrow(start=y2, end=qc2, buff=0.0, stroke_width=4, color=RED_E)

            lbl_f = MathTex(r"Q_{D^4}(y)").scale(0.6).next_to(qf_dot, DOWN)
            lbl_c = MathTex(r"Q_{2D^4}(y)").scale(0.6).next_to(qc_dot, DOWN)

            self.play(FadeIn(y_dot, scale=0.9))
            self.play(GrowArrow(err_f), FadeIn(qf_dot), Write(lbl_f))
            self.wait(0.2)
            self.play(GrowArrow(err_c), FadeIn(qc_dot), Write(lbl_c))
            self.wait(0.6)

            self.play(FadeOut(err_f), FadeOut(err_c), FadeOut(lbl_f), FadeOut(lbl_c), FadeOut(y_dot), FadeOut(qf_dot), FadeOut(qc_dot))

        note = MathTex(r"\|y-Q_{D^4}(y)\| \le \|y-Q_{2D^4}(y)\|\ \text{ (fine ≤ coarse)}").scale(0.75).to_edge(RIGHT)
        self.play(Write(note))
        self.wait(2)


class Mod2D4Demo(Scene):
    def construct(self):
        plane = make_plane()
        self.add(plane)

        title = Text("Modulo coarse lattice:  y mod 2·D⁴", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=DOWN*0.5))

        _, coarse2 = gen_lattice_points(B_COARSE, kmax=2)
        coarse = VGroup(*[Dot(p, radius=0.06, color=YELLOW, fill_opacity=1.0, z_index=2) for p in coarse2])
        self.add(coarse)

        # Notionally show one "fundamental region" proxy in projected plane: a disk centered at origin
        fr_proxy = Circle(radius=1.4, color=TEAL, fill_opacity=0.10).move_to(ORIGIN)
        self.play(Create(fr_proxy))

        rng = np.random.default_rng(19)
        for _ in range(6):
            y4 = rng.uniform(-3.0, 3.0, size=4)
            y2 = project2(y4)
            y_dot = Dot(y2, color=WHITE).set_z_index(3)
            self.play(FadeIn(y_dot, scale=0.9))

            q4 = nearest_lattice_point(B_COARSE, y4)
            q2 = project2(q4)
            q_dot = Dot(q2, color=YELLOW).set_z_index(3)
            arrow_q = Arrow(start=y2, end=q2, buff=0.0, stroke_width=4, color=YELLOW_E)

            ymod4 = y4 - q4
            ymod2 = project2(ymod4)
            ymod_dot = Dot(ymod2, color=TEAL).set_z_index(3)
            arrow_mod = Arrow(start=y2, end=ymod2, buff=0.0, stroke_width=4, color=TEAL_E)

            lbl_q = MathTex(r"Q_{2D^4}(y)").scale(0.6).next_to(q_dot, UP)
            lbl_mod = MathTex(r"y \bmod 2D^4").scale(0.6).next_to(ymod_dot, DOWN)

            self.play(GrowArrow(arrow_q), FadeIn(q_dot), Write(lbl_q))
            self.wait(0.2)
            self.play(GrowArrow(arrow_mod), FadeIn(ymod_dot), Write(lbl_mod))
            self.wait(0.7)

            self.play(FadeOut(arrow_q), FadeOut(arrow_mod), FadeOut(lbl_q), FadeOut(lbl_mod), FadeOut(y_dot), FadeOut(q_dot), FadeOut(ymod_dot))

        eq = MathTex(r"y \bmod 2D^4 \in \mathcal{V}_{2D^4}\ \text{(in 4D; proxy shown in projection)}").scale(0.72).to_edge(RIGHT)
        self.play(Write(eq))
        self.wait(2)
