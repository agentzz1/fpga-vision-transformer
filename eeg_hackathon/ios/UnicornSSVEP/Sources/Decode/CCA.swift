//
//  CCA.swift
//  SSVEP2048
//
//  Canonical Correlation Analysis (CCA) for SSVEP decoding.
//
//  Computes the largest canonical correlation between an EEG window X (N×p,
//  samples × channels) and a sinusoidal reference bank Y (N×q, samples ×
//  2*harmonics). The largest canonical correlation ρ₁ is the score used by the
//  filter-bank decoder (FBCCA) to rank candidate stimulus frequencies.
//
//  Algorithm (Contract §B.11, "Route A"):
//      1. Mean-center each column of X and Y.
//      2. Thin QR factorization of each:  X = Qx·Rx ,  Y = Qy·Ry
//         (orthonormal bases Qx ∈ ℝ^{N×p}, Qy ∈ ℝ^{N×q}). This whitens the
//         column spaces — the canonical correlations of (X, Y) are exactly the
//         singular values of M = Qxᵀ·Qy.
//      3. M = Qxᵀ·Qy   (p×q)
//      4. SVD of M; σ₁ (largest singular value) == ρ₁, the max canonical
//         correlation, clamped to [0, 1].
//
//  All linear algebra uses Accelerate / LAPACK in single precision (Float),
//  matching the contract's window layout (column-major, N×C). LAPACK is natively
//  column-major, so the contract's channel-major windows can be fed directly.
//
//  Numerical robustness: rank deficiency in X or Y (constant channels, fewer
//  unique samples than columns, duplicated references) collapses the relevant QR
//  rank. We detect tiny diagonal R magnitudes and drop the corresponding basis
//  columns so the SVD operates only on well-conditioned directions. Any LAPACK
//  failure or degenerate input returns 0 (no correlation) rather than garbage.
//
//  Created for the SSVEP2048 hackathon app.
//

import Foundation
import Accelerate

enum CCA {

    /// Largest canonical correlation between X (N×p) and Y (N×q), both
    /// **column-major** (each column a contiguous N-block, i.e. LAPACK layout).
    ///
    /// - Parameters:
    ///   - X: pointer to N*p Floats, column-major (samples × channels).
    ///   - p: number of columns of X (channels).
    ///   - Y: pointer to N*q Floats, column-major (samples × 2*harmonics).
    ///   - q: number of columns of Y.
    ///   - N: number of rows (samples) shared by X and Y.
    /// - Returns: ρ₁ ∈ [0, 1]. Returns 0 for degenerate / rank-deficient input.
    static func canonicalCorrelationMax(X: UnsafePointer<Float>, p: Int,
                                        Y: UnsafePointer<Float>, q: Int,
                                        N: Int) -> Float {
        // ---- Guard trivial / malformed dimensions -----------------------------
        // QR/SVD require at least one row, one column per matrix, and (for a thin,
        // full-rank QR) N >= p and N >= q. If there are fewer samples than columns
        // the column space cannot span all references; we still proceed but the
        // rank-deficiency handling below will trim the extra directions.
        guard N > 1, p > 0, q > 0 else { return 0 }

        // ---- Build mean-centered, orthonormal bases via QR --------------------
        // orthonormalBasis returns Q (N×r) flattened column-major along with its
        // effective numerical rank r (r <= original column count).
        guard let (qx, rx) = orthonormalBasis(matrix: X, rows: N, cols: p),
              let (qy, ry) = orthonormalBasis(matrix: Y, rows: N, cols: q),
              rx > 0, ry > 0 else {
            // A zero-rank basis means a fully constant (DC-only) block — no
            // meaningful canonical structure. Score 0.
            return 0
        }

        // ---- Cross matrix M = Qxᵀ · Qy   (rx × ry) ----------------------------
        // Canonical correlations of (X, Y) == singular values of M, because Qx, Qy
        // are orthonormal whitenings of the two column spaces.
        var M = [Float](repeating: 0, count: rx * ry)
        // cblas_sgemm with column-major ordering:
        //   M(rx×ry) = Qxᵀ(rx×N) · Qy(N×ry)
        qx.withUnsafeBufferPointer { qxp in
            qy.withUnsafeBufferPointer { qyp in
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            Int32(rx), Int32(ry), Int32(N),
                            1.0,
                            qxp.baseAddress!, Int32(N),   // Qx is N×rx, lda = N
                            qyp.baseAddress!, Int32(N),   // Qy is N×ry, ldb = N
                            0.0,
                            &M, Int32(rx))                // M is rx×ry, ldc = rx
            }
        }

        // ---- Largest singular value of M == ρ₁ --------------------------------
        let sigma1 = largestSingularValue(matrix: &M, rows: rx, cols: ry)

        // Canonical correlations are bounded by [0, 1]; floating-point round-off
        // can nudge slightly past 1. Clamp for a well-behaved score.
        return min(max(sigma1, 0), 1)
    }

    // MARK: - QR orthonormal basis with rank detection

    /// Mean-centers each column of `matrix` (rows×cols, column-major), then
    /// computes a thin orthonormal basis Q (rows×r) of its column space via
    /// LAPACK QR (sgeqrf + sorgqr). `r` is the numerical rank, determined by
    /// thresholding the magnitudes of the R diagonal.
    ///
    /// - Returns: (Q flattened column-major length rows*r, r), or nil on failure.
    private static func orthonormalBasis(matrix src: UnsafePointer<Float>,
                                         rows: Int, cols: Int) -> ([Float], Int)? {
        // Working copy A (rows×cols), column-major. LAPACK overwrites it in place.
        var A = [Float](repeating: 0, count: rows * cols)
        for c in 0..<cols {
            // Compute column mean.
            var mean: Float = 0
            let base = c * rows
            for r in 0..<rows { mean += src[base + r] }
            mean /= Float(rows)
            // Subtract mean (centering) into the working copy.
            for r in 0..<rows { A[base + r] = src[base + r] - mean }
        }

        var m = __LAPACK_int(rows)
        var n = __LAPACK_int(cols)
        var lda = __LAPACK_int(rows)
        var info: __LAPACK_int = 0

        // tau holds the Householder scalar factors (length min(m,n)).
        let k = min(rows, cols)
        var tau = [Float](repeating: 0, count: k)

        // ---- Workspace query for sgeqrf ---------------------------------------
        var workQuery: Float = 0
        var lwork: __LAPACK_int = -1
        sgeqrf_(&m, &n, &A, &lda, &tau, &workQuery, &lwork, &info)
        guard info == 0 else { return nil }
        lwork = __LAPACK_int(workQuery)
        if lwork < 1 { lwork = 1 }
        var work = [Float](repeating: 0, count: Int(lwork))

        // ---- QR factorization: A overwritten with R (upper) + Householder vecs -
        sgeqrf_(&m, &n, &A, &lda, &tau, &work, &lwork, &info)
        guard info == 0 else { return nil }

        // ---- Numerical rank from |R[i,i]| -------------------------------------
        // R is stored in the upper triangle of A (column-major). The diagonal
        // entry R[i,i] lives at A[i*rows + i]. A column whose pivot magnitude is
        // negligible relative to the largest pivot indicates a linearly dependent
        // (rank-deficient) direction.
        var maxDiag: Float = 0
        for i in 0..<k {
            let d = abs(A[i * rows + i])
            if d > maxDiag { maxDiag = d }
        }
        // Relative rank threshold tied to single-precision machine epsilon
        // (`Float.ulpOfOne` ≈ 1.19e-7), scaled by the larger matrix dimension —
        // the standard LAPACK-style numerical-rank tolerance. Avoids the previous
        // magic 1e-6 that was not tied to FLT_EPSILON. If everything is ~0 the
        // matrix is effectively constant → rank 0.
        let tol = maxDiag * Float(max(rows, cols)) * Float.ulpOfOne
        var rank = 0
        for i in 0..<k where abs(A[i * rows + i]) > tol { rank += 1 }
        guard rank > 0 else { return ([], 0) }

        // ---- Generate the explicit orthonormal Q (rows × rank) ----------------
        // sorgqr builds the first `rank` columns of Q from the Householder
        // reflectors. We request only `rank` columns so trailing dependent
        // directions are discarded.
        var ncolsQ = __LAPACK_int(rank)
        var kRef = __LAPACK_int(k)            // # of reflectors available
        // sorgqr requires the supplied k <= number of generated columns; clamp.
        if kRef > ncolsQ { kRef = ncolsQ }

        var workQuery2: Float = 0
        var lwork2: __LAPACK_int = -1
        sorgqr_(&m, &ncolsQ, &kRef, &A, &lda, &tau, &workQuery2, &lwork2, &info)
        guard info == 0 else { return nil }
        lwork2 = __LAPACK_int(workQuery2)
        if lwork2 < 1 { lwork2 = 1 }
        var work2 = [Float](repeating: 0, count: Int(lwork2))

        sorgqr_(&m, &ncolsQ, &kRef, &A, &lda, &tau, &work2, &lwork2, &info)
        guard info == 0 else { return nil }

        // A now holds Q in its first `rank` columns (column-major, leading dim
        // = rows). Extract them into a tightly-packed rows*rank buffer.
        var Q = [Float](repeating: 0, count: rows * rank)
        for c in 0..<rank {
            let srcBase = c * rows
            let dstBase = c * rows
            for r in 0..<rows { Q[dstBase + r] = A[srcBase + r] }
        }
        return (Q, rank)
    }

    // MARK: - Largest singular value

    /// Largest singular value of `matrix` (rows×cols, column-major), via LAPACK
    /// SVD (sgesdd, singular values only — jobz = 'N'). Returns 0 on failure.
    private static func largestSingularValue(matrix: inout [Float],
                                             rows: Int, cols: Int) -> Float {
        guard rows > 0, cols > 0 else { return 0 }

        var jobz: CChar = 78 // ASCII 'N' — singular values only, no U/V vectors.
        var m = __LAPACK_int(rows)
        var n = __LAPACK_int(cols)
        var lda = __LAPACK_int(rows)
        let minDim = min(rows, cols)
        var s = [Float](repeating: 0, count: minDim)

        // U and VT are not referenced for jobz='N', but LAPACK still wants valid
        // (possibly size-1) pointers and leading dimensions.
        var u: Float = 0
        var ldu = __LAPACK_int(1)
        var vt: Float = 0
        var ldvt = __LAPACK_int(1)

        // sgesdd integer workspace: 8*min(m,n).
        var iwork = [__LAPACK_int](repeating: 0, count: 8 * minDim)
        var info: __LAPACK_int = 0

        // ---- Workspace query ---------------------------------------------------
        var workQuery: Float = 0
        var lwork: __LAPACK_int = -1
        sgesdd_(&jobz, &m, &n, &matrix, &lda, &s, &u, &ldu, &vt, &ldvt,
                &workQuery, &lwork, &iwork, &info)
        guard info == 0 else { return 0 }
        lwork = __LAPACK_int(workQuery)
        if lwork < 1 { lwork = 1 }
        var work = [Float](repeating: 0, count: Int(lwork))

        // ---- Compute singular values ------------------------------------------
        sgesdd_(&jobz, &m, &n, &matrix, &lda, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &iwork, &info)
        guard info == 0 else { return 0 }

        // LAPACK returns singular values in descending order; s[0] is the largest.
        return s.first ?? 0
    }
}

// MARK: - Convenience [[Double]] entry point

extension CCA {
    /// Convenience wrapper matching the task description's `[[Double]]` shape.
    ///
    /// `X` is samples × channels (each inner array = one sample of all channels),
    /// `Y` is samples × (2*harmonics) (each inner array = one sample of all
    /// reference signals). Both must share the same number of rows (samples).
    ///
    /// Internally repacks to column-major Float and delegates to
    /// `canonicalCorrelationMax(X:p:Y:q:N:)`. Returns 0 for mismatched or empty
    /// input. Provided for ergonomic / test use; the hot path in FBCCA should
    /// call the pointer-based core directly.
    static func maxCanonicalCorrelation(X: [[Double]], Y: [[Double]]) -> Double {
        let n = X.count
        guard n > 1, n == Y.count, let p = X.first?.count, p > 0,
              let q = Y.first?.count, q > 0 else { return 0 }

        // Validate uniform row widths to avoid ragged-array reads.
        guard X.allSatisfy({ $0.count == p }), Y.allSatisfy({ $0.count == q }) else {
            return 0
        }

        // Repack row-major [[Double]] → column-major [Float].
        var xCol = [Float](repeating: 0, count: n * p)
        var yCol = [Float](repeating: 0, count: n * q)
        for r in 0..<n {
            let xr = X[r]
            for c in 0..<p { xCol[c * n + r] = Float(xr[c]) }
            let yr = Y[r]
            for c in 0..<q { yCol[c * n + r] = Float(yr[c]) }
        }

        let rho = xCol.withUnsafeBufferPointer { xp in
            yCol.withUnsafeBufferPointer { yp in
                canonicalCorrelationMax(X: xp.baseAddress!, p: p,
                                        Y: yp.baseAddress!, q: q, N: n)
            }
        }
        return Double(rho)
    }
}
