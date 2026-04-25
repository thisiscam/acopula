"""Test alternative approaches for nested Rosenblatt sampling."""

import jax
import jax.numpy as jnp
from jax import random as jrandom
jax.config.update("jax_enable_x64", True)

from acopula.core import copula
import jet_array

@copula
class Clayton:
    theta: float
    def generator(self, t):
        return (1.0 + t) ** (-1.0 / self.theta)

# =====================================================
# Approach 2: Hierarchical Conditional CDF Factorization
# =====================================================

def test_hierarchical_factorization():
    """Test: for leaves within the same group, the Rosenblatt conditional CDF
    equals the within-group conditional (using just the group's generator).

    For a nested copula C(u1,u2,u3,u4) = psi0(psi0inv(psi1(t1)) + psi0inv(psi1(t2)))
    where t1 = psi1inv(u1)+psi1inv(u2), t2 = psi1inv(u3)+psi1inv(u4),
    the conditional CDF of u2 given u1 should be just the within-group formula:
      psi1'(t_new) / psi1'(t_prev)
    because u3,u4 marginalize out.
    """
    theta0, theta1 = 2.0, 5.0
    root = Clayton(theta0)
    child = Clayton(theta1)

    psi0 = root.generator
    psi0_inv = root.generator_inv
    psi1 = child.generator
    psi1_inv = child.generator_inv

    u1, u2, u3, u4 = 0.3, 0.6, 0.4, 0.7

    # Full CDF
    def C(u1_, u2_, u3_, u4_):
        t1 = psi1_inv(u1_) + psi1_inv(u2_)
        t2 = psi1_inv(u3_) + psi1_inv(u4_)
        s = psi0_inv(psi1(t1)) + psi0_inv(psi1(t2))
        return psi0(s)

    # --- Flat AD approach: F_{2|1}(u2) = dC(u1,u2,1,1)/du1 / dC(u1,1,1,1)/du1 ---
    def C_u1_u2(u1_, u2_):
        return C(u1_, u2_, 1.0, 1.0)

    dC_du1 = jax.grad(C_u1_u2, argnums=0)
    numerator = dC_du1(u1, u2)
    denominator = dC_du1(u1, 1.0)
    cond_flat = numerator / denominator

    # --- Within-group approach: psi1'(t_new)/psi1'(t_prev) ---
    t_prev = psi1_inv(u1)
    t_new = t_prev + psi1_inv(u2)

    series_in = jnp.zeros(1).at[0].set(1.0)
    _, s_new = jet_array.jet(psi1, (t_new,), (series_in,))
    _, s_old = jet_array.jet(psi1, (t_prev,), (series_in,))
    cond_within = s_new[0] / s_old[0]

    print(f"C(u2|u1) via flat AD:       {cond_flat:.12f}")
    print(f"C(u2|u1) via within-group:  {cond_within:.12f}")
    print(f"Match: {jnp.allclose(cond_flat, cond_within, atol=1e-10)}")
    print()

    # --- Now test: F_{3|1,2}(u3) --- cross-group conditional
    # This is where the hierarchical factorization gets interesting.
    # F_{3|1,2}(u3) = d^2 C(u1,u2,u3,1)/du1 du2 / d^2 C(u1,u2,1,1)/du1 du2

    def C_123(u1_, u2_, u3_):
        return C(u1_, u2_, u3_, 1.0)

    d2C_du1du2_num = jax.grad(jax.grad(C_123, argnums=0), argnums=1)
    d2C_du1du2_den_fn = jax.grad(jax.grad(C_u1_u2, argnums=0), argnums=1)

    num3 = d2C_du1du2_num(u1, u2, u3)
    den3 = d2C_du1du2_den_fn(u1, u2)
    cond_u3_flat = num3 / den3

    # Hierarchical decomposition for cross-group:
    # The conditional CDF of u3 given u1,u2 can be seen as:
    # The "group value" for group 1 is G1 = psi1(psi1inv(u1)+psi1inv(u2))
    # The root copula sees (G1, G2) where G2 = psi1(psi1inv(u3)+psi1inv(u4))
    # Since u4 is marginalized (=1, so psi1inv(1)=0), G2 = u3.
    # So F_{3|1,2}(u3) = conditional of u3 given G1 under root copula.
    #
    # But the conditional involves the mixed partial d^2 C / du1 du2, not just G1.
    # The key insight: the mixed partial factors!
    #
    # d^2 C / du1 du2 = psi0''(S) * (psi0inv)'(G1) * h'(t1) * (psi1inv)'(u1)
    #                   * (psi0inv)'(G1) * h'(t1) * (psi1inv)'(u2)
    #   Wait, this isn't right either. Let me compute properly.

    # Group 1 value
    t1_val = psi1_inv(u1) + psi1_inv(u2)
    G1 = psi1(t1_val)

    # At the root level, the conditional of the second group given the first:
    # F_{G2|G1}(G2) = psi0'(psi0inv(G1) + psi0inv(G2)) / psi0'(psi0inv(G1))
    # With G2 = u3 (since u4 marginalized):
    S_old = psi0_inv(G1)
    S_new = psi0_inv(G1) + psi0_inv(u3)

    series_in_1 = jnp.zeros(1).at[0].set(1.0)
    _, sr_new = jet_array.jet(psi0, (S_new,), (series_in_1,))
    _, sr_old = jet_array.jet(psi0, (S_old,), (series_in_1,))
    cond_u3_hierarchical = sr_new[0] / sr_old[0]

    print(f"C(u3|u1,u2) via flat AD:        {cond_u3_flat:.12f}")
    print(f"C(u3|u1,u2) via hierarchical:   {cond_u3_hierarchical:.12f}")
    print(f"Match: {jnp.allclose(cond_u3_flat, cond_u3_hierarchical, atol=1e-10)}")
    print()

    # --- Now test: F_{4|1,2,3}(u4) --- second leaf in second group
    # F_{4|1,2,3} = d^3 C(u1,u2,u3,u4)/du1 du2 du3 / d^3 C(u1,u2,u3,1)/du1 du2 du3

    d3C_num = jax.grad(jax.grad(jax.grad(C, argnums=0), argnums=1), argnums=2)
    d3C_den = jax.grad(jax.grad(jax.grad(C_123, argnums=0), argnums=1), argnums=2)

    num4 = d3C_num(u1, u2, u3, u4)
    den4 = d3C_den(u1, u2, u3)
    cond_u4_flat = num4 / den4

    # Hierarchical approach for u4 given u1,u2,u3:
    # Group 2 has leaves u3, u4. We've placed u3, now place u4.
    # Within group 2: conditional of u4 given u3 in group 2's copula
    # BUT the root conditional also changes because G2 changes from u3 to psi1(t2).
    #
    # The factorization should be:
    # F_{4|1,2,3}(u4) = [within-group factor] * [root adjustment]?
    #
    # Actually, let me think about this differently using the Bell polynomial insight.
    # The mixed partial d^3 C / du1 du2 du3 evaluated at (u1,u2,u3,u4) can be written
    # using the Bell polynomial decomposition:
    #   = sum_k beta[k] * psi0^{(k)}(S)
    # where S = psi0inv(G1) + psi0inv(G2), and beta encodes the contributions from
    # each group's internal structure.
    #
    # For the ratio F_{4|1,2,3}(u4):
    # numerator = sum_k beta'[k] * psi0^{(k)}(S_new) where S_new uses G2(u3,u4)
    # denominator = sum_k beta[k] * psi0^{(k)}(S_old) where S_old uses G2(u3)
    #
    # The beta coefficients change between num and den because the group 2 alpha changes.
    # This is exactly what the current code does!

    # Let me check: can we STILL decompose into within-group * between-group?
    # For the within-group part of group 2:
    t2_prev = psi1_inv(u3)
    t2_new = t2_prev + psi1_inv(u4)
    _, s2_new = jet_array.jet(psi1, (t2_new,), (series_in_1,))
    _, s2_old = jet_array.jet(psi1, (t2_prev,), (series_in_1,))
    within_group2 = s2_new[0] / s2_old[0]

    # For the between-group factor:
    G2_old = psi1(t2_prev)  # = u3
    G2_new = psi1(t2_new)   # = psi1(psi1inv(u3)+psi1inv(u4))

    # Root-level conditional changes from order=1 (G1 processed) to still order=1
    # but G2 value changes.
    # The root sees 2 group values. After processing group 1 fully (order=2 from u1,u2 perspective,
    # but at the root level, group 1 contributes as a single entity with inner order 2).

    # Hmm, the factorization is NOT simply a product of within-group and between-group.
    # Because the root generator's higher derivatives interact with the inner structure.

    # But maybe for the SPECIAL CASE where we process groups sequentially,
    # after fully processing group 1, the conditional for group 2 leaves
    # can be decomposed differently.

    print(f"C(u4|u1,u2,u3) via flat AD:     {cond_u4_flat:.12f}")
    print(f"Within-group2 factor:            {within_group2:.12f}")
    print(f"Ratio (flat/within):             {cond_u4_flat / within_group2:.12f}")
    print()

    return True


# =====================================================
# Approach 4: Marshall-Olkin + Rosenblatt Hybrid
# =====================================================

def test_mo_rosenblatt_hybrid():
    """Test: Use MO for root frailty, Rosenblatt within groups.

    The idea:
    1. Sample root frailty V0 from the root mixing distribution (using Post-Widder/Talbot).
    2. For each group g with generator psi_g, the conditional Laplace transform is:
       psi_tilde_g(t) = exp(-V0 * psi0inv(psi_g(t)))
       This defines the group's conditional distribution.
    3. Use simple (non-nested) Rosenblatt WITHIN each group using psi_tilde_g.
       Since psi_tilde_g is a valid generator, the Rosenblatt formula applies directly:
       C_{j|1..j-1}(u_j) = psi_tilde_g^{(j-1)}(t_new) / psi_tilde_g^{(j-1)}(t_prev)

    This avoids:
    - Needing V01 conditional frailties
    - Complex nested Bell polynomial machinery for sampling
    - The 3+ nesting depth limitation

    The key challenge: computing psi_tilde_g^{(k)}(t) via jet.
    """
    theta0, theta1 = 2.0, 5.0
    root = Clayton(theta0)
    child = Clayton(theta1)

    psi0 = root.generator
    psi0_inv = root.generator_inv
    psi1 = child.generator
    psi1_inv = child.generator_inv

    u1, u2, u3, u4 = 0.3, 0.6, 0.4, 0.7

    # Full CDF
    def C(u1_, u2_, u3_, u4_):
        t1 = psi1_inv(u1_) + psi1_inv(u2_)
        t2 = psi1_inv(u3_) + psi1_inv(u4_)
        s = psi0_inv(psi1(t1)) + psi0_inv(psi1(t2))
        return psi0(s)

    # First, let's verify the MO+Rosenblatt approach produces valid samples.
    # For a FIXED V0, the modified generator is:
    # psi_tilde(t) = exp(-V0 * psi0inv(psi1(t)))

    # Let's check: does the standard Rosenblatt formula applied to psi_tilde
    # give the correct conditional CDFs?

    # For a fixed V0, the copula within a group is:
    # C_group(u1,...,um; V0) = exp(-V0 * S) where S = psi0inv(psi1(sum_j psi1inv(uj)))
    #   wait, actually:
    # C_group(u1,...,um; V0) = psi_tilde(psi1inv(u1) + ... + psi1inv(um))
    #                        = exp(-V0 * psi0inv(psi1(psi1inv(u1)+...+psi1inv(um))))
    # But this is NOT the same as the marginal group copula psi1(psi1inv(u1)+...+psi1inv(um)).
    #
    # The MO approach gives us U_j = psi_tilde(-log(X_j)/V0) for iid X_j ~ Exp(1),
    # and then the joint is C(u1,...,ud) = E[prod_g prod_j psi_tilde_g(-log(X_j)/V0)].
    #
    # So for SAMPLING, the MO approach works at the root level:
    # 1. Sample V0 from root mixing distribution
    # 2. For group g: need to sample from the CONDITIONAL copula given V0
    #    - If we could sample V_g from the conditional mixing dist of group g given V0,
    #      we'd be done. But that's the V01 problem.
    #    - Instead, use Rosenblatt WITHIN the group, treating psi_tilde_g as the generator.

    # The question is: does psi_tilde_g(t) = exp(-V0 * psi0inv(psi1(t)))
    # define a valid Archimedean generator?
    # - psi_tilde_g(0) = exp(-V0 * psi0inv(psi1(0))) = exp(-V0 * psi0inv(1)) = exp(0) = 1.
    #   Wait: psi0inv(1)? For Clayton: psi0(t) = (1+t)^(-1/theta), so psi0inv(u) = u^(-theta) - 1.
    #   psi0inv(1) = 0. So psi_tilde_g(0) = exp(0) = 1. Good.
    # - psi_tilde_g(inf) = exp(-V0 * inf) = 0. Good.
    # - psi_tilde_g is decreasing and completely monotone?
    #   The composition psi0inv(psi1(t)) is increasing (psi1 decreasing, psi0inv decreasing).
    #   So -V0 * psi0inv(psi1(t)) is decreasing. And exp of decreasing is decreasing.
    #   Complete monotonicity: exp(-V0 * h(t)) where h is a Bernstein function...
    #   This IS completely monotone (Laplace transform of a positive measure).

    # Great! So psi_tilde_g is a valid generator. Now the Rosenblatt within group g:
    # For the first leaf: u_1 = psi_tilde_g^{-1}(V0_sample_or_something)?
    # No wait. We're not doing MO within the group. We're using Rosenblatt.
    # The Rosenblatt for a simple (non-nested) copula with generator psi_tilde:
    # Sample v_1 ~ U(0,1), set u_1 = v_1
    # Sample v_2 ~ U(0,1), find u_2 such that psi_tilde'(t_new)/psi_tilde'(t_old) = v_2
    # where t_old = psi_tilde_inv(u_1), t_new = t_old + psi_tilde_inv(u_2).

    # But we need psi_tilde_inv!
    # psi_tilde(t) = exp(-V0 * psi0inv(psi1(t)))
    # psi_tilde_inv(u) = psi1_inv(psi0((-log(u))/V0))
    # That's easy! It's just a composition.

    V0 = 1.5  # Fixed frailty for testing

    def psi_tilde(t):
        return jnp.exp(-V0 * psi0_inv(psi1(t)))

    def psi_tilde_inv(u):
        return psi1_inv(psi0(-jnp.log(u) / V0))

    # Verify inverse
    test_t = 0.5
    print(f"psi_tilde(test_t) = {psi_tilde(test_t):.12f}")
    print(f"psi_tilde_inv(psi_tilde(test_t)) = {psi_tilde_inv(psi_tilde(test_t)):.12f}")
    print(f"Inverse works: {jnp.allclose(psi_tilde_inv(psi_tilde(test_t)), test_t, atol=1e-10)}")
    print()

    # Verify that psi_tilde gives the right copula CDF.
    # C_tilde(u1,u2) = psi_tilde(psi_tilde_inv(u1) + psi_tilde_inv(u2))
    # should equal C_conditional(u1,u2|V0) = exp(-V0 * (psi0inv(psi1(psi1inv(u1)+psi1inv(u2)))))
    # Hmm wait, psi_tilde_inv(u) is NOT psi1inv(u). Let me reconsider.

    # The copula C_tilde with generator psi_tilde has:
    # C_tilde(u1,...,um) = psi_tilde(sum_j psi_tilde_inv(u_j))
    # But the actual conditional copula given V0 (from the MO decomposition) is:
    # The leaves in a group have U_j = psi1(-log(X_j)/V_g) where V_g ~ F_{V_g|V0}.
    #
    # Actually, the correct conditional copula given V0 for a single group is:
    # P(U_1<=u_1,...,U_m<=u_m | V0) where the U_j are obtained by:
    #   For each group: sample V_g | V0, then U_j = psi_g(-log(X_j)/V_g)
    # This conditional copula is NOT simply psi_tilde applied with Archimedean structure.
    #
    # The nested copula C(u1,...,ud) = E_{V0}[prod_g exp(-V0 * h_g(u_{g1},...,u_{gm}))]
    # where h_g(u_{g1},...,u_{gm}) = psi0inv(psi_g(sum_j psi_g_inv(u_gj)))
    # So the conditional on V0 is:
    # C(u|V0) = prod_g exp(-V0 * h_g(u_g))
    #
    # For a single group g:
    # C_g(u_{g1},...,u_{gm}|V0) = exp(-V0 * psi0inv(psi_g(sum_j psi_g_inv(u_gj))))
    #                             = psi_tilde_g(sum_j psi_g_inv(u_gj))
    #
    # So C_g conditional on V0 uses generator psi_tilde BUT with inverse
    # taken wrt the INNER generator psi_g, not psi_tilde!
    # C_g(u|V0) = psi_tilde_g(sum_j psi_g_inv(u_j))
    # This is NOT a standard Archimedean copula with generator psi_tilde_g,
    # because a standard AC would use psi_tilde_inv instead of psi_g_inv.
    #
    # Unless psi_tilde_inv = psi_g_inv? Let's check:
    # psi_tilde_inv(u) = psi_g_inv(psi0(-log(u)/V0))
    # psi_g_inv(u) = psi_g_inv(u)
    # These are NOT the same.

    # So the conditional copula given V0 is NOT Archimedean with generator psi_tilde!
    # It's a different beast: C_g(u|V0) = psi_tilde(sum_j psi_g_inv(u_j)).
    # This uses a MIXED pair of generators (psi_tilde for outer, psi_g for inner).

    # BUT we can still do Rosenblatt on this:
    # C_g(u_1,...,u_m|V0) = psi_tilde(sum_j psi_g_inv(u_j))
    # The conditional CDF of u_j given u_{<j}:
    # = d^{j-1}/du_1...du_{j-1} psi_tilde(S_j) / d^{j-1}/du_1...du_{j-1} psi_tilde(S_{j-1})
    # where S_k = sum_{i=1}^k psi_g_inv(u_i)
    #
    # Each du_i contributes a factor of (psi_g_inv)'(u_i) via chain rule.
    # So d^{j-1}/du_1...du_{j-1} psi_tilde(S_j)
    #    = psi_tilde^{(j-1)}(S_j) * prod_{i=1}^{j-1} (psi_g_inv)'(u_i)
    #
    # And similarly for S_{j-1}.
    # The (psi_g_inv)'(u_i) terms cancel in the ratio!
    # So F_{j|<j}(u_j|V0) = psi_tilde^{(j-1)}(S_j) / psi_tilde^{(j-1)}(S_{j-1})
    # where S_k = sum_{i=1}^k psi_g_inv(u_i)
    #
    # This is EXACTLY the standard Rosenblatt formula, just with psi_tilde as generator
    # and psi_g_inv for computing the S values!

    # Let's verify this numerically.
    # For V0=1.5, the conditional copula for group 1 is:
    # C(u1,u2|V0) = psi_tilde(psi1inv(u1) + psi1inv(u2))
    # F_{2|1}(u2|u1,V0) = psi_tilde'(S2) / psi_tilde'(S1)
    # where S1 = psi1inv(u1), S2 = psi1inv(u1) + psi1inv(u2)

    S1 = psi1_inv(u1)
    S2 = S1 + psi1_inv(u2)

    series_in_1 = jnp.zeros(1).at[0].set(1.0)
    _, st_new = jet_array.jet(psi_tilde, (S2,), (series_in_1,))
    _, st_old = jet_array.jet(psi_tilde, (S1,), (series_in_1,))
    cond_hybrid = st_new[0] / st_old[0]

    # Compare with computing the conditional directly from the full conditional copula
    def C_cond_g1(u1_, u2_):
        return psi_tilde(psi1_inv(u1_) + psi1_inv(u2_))

    dC_cond = jax.grad(C_cond_g1, argnums=0)
    num = dC_cond(u1, u2)
    den = dC_cond(u1, 1.0)
    cond_direct = num / den

    print("=== MO+Rosenblatt Hybrid: Within-group conditional given V0 ===")
    print(f"F(u2|u1,V0) via jet on psi_tilde:  {cond_hybrid:.12f}")
    print(f"F(u2|u1,V0) via direct AD:         {cond_direct:.12f}")
    print(f"Match: {jnp.allclose(cond_hybrid, cond_direct, atol=1e-10)}")
    print()

    # Now test inversion via bisection
    # Given v2 ~ U(0,1), find u2 such that psi_tilde'(S2)/psi_tilde'(S1) = v2
    v2_target = 0.45

    def cond_cdf(u2_):
        S2_ = S1 + psi1_inv(u2_)
        _, st_n = jet_array.jet(psi_tilde, (S2_,), (series_in_1,))
        _, st_o = jet_array.jet(psi_tilde, (S1,), (series_in_1,))
        return st_n[0] / st_o[0]

    # Simple bisection
    lo, hi = 1e-14, 1.0 - 1e-14
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if cond_cdf(mid) < v2_target:
            lo = mid
        else:
            hi = mid
    u2_found = (lo + hi) / 2.0
    print(f"Bisection: v2_target={v2_target}, u2_found={u2_found:.12f}")
    print(f"Verification: cond_cdf(u2_found) = {cond_cdf(u2_found):.12f}")
    print()


# =====================================================
# Approach 1: Forward-mode mixed partials via nested JVP
# =====================================================

def test_forward_mode_mixed_partials():
    """Test: compute mixed partial d^k C / du_1...du_k using nested jvp calls.

    For k mixed partials of a scalar function, we need k nested jvp calls.
    Each jvp is a forward-mode pass. For scalar output, forward-mode has the
    same cost as reverse-mode per pass. So k nested jvp = k nested grad.

    However, for the Rosenblatt we don't need ALL mixed partials — we need
    d^{j-1} C / du_1...du_{j-1} evaluated at (u1,...,u_j, 1,...,1) and
    (u1,...,u_{j-1}, 1,...,1). The jet approach computes this more efficiently
    because the dependence on u1,...,u_{j-1} is only through S = sum psi_inv(u_i).

    Let me verify the jet approach matches nested jvp for correctness.
    """
    theta0, theta1 = 2.0, 5.0
    root = Clayton(theta0)
    child = Clayton(theta1)

    psi0 = root.generator
    psi0_inv = root.generator_inv
    psi1 = child.generator
    psi1_inv = child.generator_inv

    u1, u2, u3 = 0.3, 0.6, 0.4

    def C(u1_, u2_, u3_):
        t1 = psi1_inv(u1_) + psi1_inv(u2_)
        s = psi0_inv(psi1(t1)) + psi0_inv(u3_)
        return psi0(s)

    # Method 1: Nested jax.grad (reverse mode)
    d2C_grad = jax.grad(jax.grad(C, argnums=0), argnums=1)
    val_grad = d2C_grad(u1, u2, u3)

    # Method 2: Nested jax.jvp (forward mode)
    # d/du1 C = jvp(C, (u1,u2,u3), (1,0,0))[1]
    # d^2/du1 du2 C = jvp(d/du1 C, (u1,u2,u3), (0,1,0))[1]

    def dC_du1(u1_, u2_, u3_):
        primals = (u1_, u2_, u3_)
        tangents = (1.0, 0.0, 0.0)
        _, tangent_out = jax.jvp(C, primals, tangents)
        return tangent_out

    primals2 = (u1, u2, u3)
    tangents2 = (0.0, 1.0, 0.0)
    _, val_jvp = jax.jvp(dC_du1, primals2, tangents2)

    print("=== Forward-mode vs Reverse-mode mixed partials ===")
    print(f"d^2C/du1du2 via nested grad: {val_grad:.12f}")
    print(f"d^2C/du1du2 via nested jvp:  {val_jvp:.12f}")
    print(f"Match: {jnp.allclose(val_grad, val_jvp, atol=1e-10)}")
    print()

    # Method 3: Using jet (Taylor-mode AD) — what the current code does
    # The mixed partial d^k C / du_1...du_k is computed via:
    # Because all u_i enter through S = sum psi_inv(u_i), the mixed partial
    # = psi^{(k)}(S) * prod_i (psi_inv)'(u_i)
    # For the non-nested case. For nested, it's more complex (Bell polynomials).

    # Compare computational cost
    import time

    def C_full(u1_, u2_, u3_, u4_, u5_, u6_):
        t1 = psi1_inv(u1_) + psi1_inv(u2_) + psi1_inv(u3_)
        t2 = psi1_inv(u4_) + psi1_inv(u5_) + psi1_inv(u6_)
        s = psi0_inv(psi1(t1)) + psi0_inv(psi1(t2))
        return psi0(s)

    # Time nested grad (5 levels)
    grad5 = jax.grad(jax.grad(jax.grad(jax.grad(jax.grad(
        C_full, argnums=0), argnums=1), argnums=2), argnums=3), argnums=4)

    # JIT and warm up
    grad5_jit = jax.jit(grad5)
    _ = grad5_jit(0.3, 0.6, 0.4, 0.7, 0.5, 0.8)

    t0 = time.time()
    for _ in range(100):
        _ = grad5_jit(0.3, 0.6, 0.4, 0.7, 0.5, 0.8).block_until_ready()
    t_grad = (time.time() - t0) / 100

    print(f"Nested grad (5 levels, 6 vars): {t_grad*1000:.3f} ms per call")
    print(f"(Note: nested jvp would have similar cost for scalar output)")
    print()


# =====================================================
# Approach 3: Incremental Bell polynomial updates
# =====================================================

def test_incremental_bell():
    """Test: can we incrementally update beta coefficients when adding one more leaf?

    When going from Rosenblatt step j-1 to step j (adding leaf j), the beta
    coefficients change. For a leaf within a group, the group's alpha polynomial
    changes (it gains one more convolution factor). The Cauchy product with
    other groups' alphas then needs updating.

    Key insight: if alpha_old has order k and we add one more leaf (trivial alpha [0,1]),
    the new alpha is convolve(alpha_old, [0,1]) = alpha_old shifted by 1 position.
    So alpha_new[i] = alpha_old[i-1] for i >= 1, alpha_new[0] = 0.

    For the Cauchy product: beta_new = convolve(beta_prior, alpha_new)
    where beta_prior is the product of all OTHER groups' alphas (unchanged).
    So beta_new = convolve(beta_prior, shift(alpha_old)).

    This means we DON'T need to recompute the full Cauchy product from scratch!
    We just convolve beta_prior with the shifted alpha.
    """
    # This is an optimization of the existing Bell polynomial approach,
    # not a fundamentally different algorithm. The savings are:
    # - Avoid recomputing alpha from scratch (save polynomial powering)
    # - Avoid full Cauchy product (just one convolution with updated alpha)

    # However, the alpha polynomial for an internal node changes non-trivially
    # because the h-Taylor coefficients depend on t_child which changes.
    # So we can't just shift — we need to recompute h_p at the new t_child.
    # The polynomial powering step IS the bottleneck (O(d_c^2) per group).

    # For DIRECT leaves (no group structure), the shift trick works perfectly.
    # For grouped leaves, we save the Cauchy product recomputation but not
    # the polynomial powering.

    print("=== Incremental Bell polynomial updates ===")
    print("For direct leaves: O(d) per step via shift trick (alpha_new = shift(alpha_old))")
    print("For grouped leaves: saves Cauchy product but not polynomial powering")
    print("Net savings: moderate for mixed leaf/group trees, minimal for pure group trees")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("APPROACH 2: Hierarchical Conditional CDF Factorization")
    print("=" * 60)
    test_hierarchical_factorization()

    print("=" * 60)
    print("APPROACH 4: MO + Rosenblatt Hybrid")
    print("=" * 60)
    test_mo_rosenblatt_hybrid()

    print("=" * 60)
    print("APPROACH 1: Forward-mode mixed partials")
    print("=" * 60)
    test_forward_mode_mixed_partials()

    print("=" * 60)
    print("APPROACH 3: Incremental Bell polynomial updates")
    print("=" * 60)
    test_incremental_bell()
