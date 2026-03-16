# ======================================
# KNN Pseudo-code Section
# ======================================

st.markdown("---")
st.header("K-Nearest Neighbors (KNN)")

# ✅ CORRIGÉ : color: #1a1a1a ajouté pour forcer le texte foncé sur fond clair
st.markdown("""
<div style="background-color:#fef9e7; border:1px solid #d4ac0d; border-radius:6px; padding:20px; font-family: Arial, sans-serif; color: #1a1a1a;">
<b>Pseudo-code of K-Nearest Neighbors (KNN)</b>
<hr style="border:0.5px solid #d4ac0d; margin: 10px 0;">
<b>Input:</b><br>
&nbsp;&nbsp;&nbsp;Training set <i>A</i> = {(<b>x</b><sup>(1)</sup>, <b>y</b><sup>(1)</sup>), (<b>x</b><sup>(2)</sup>, <b>y</b><sup>(2)</sup>), … , (<b>x</b><sup>(n)</sup>, <b>y</b><sup>(n)</sup>)}<br>
&nbsp;&nbsp;&nbsp;Instance to classify <b>x</b><br>
&nbsp;&nbsp;&nbsp;Number of neighbors <b>K</b><br><br>
<b>Output:</b><br>
&nbsp;&nbsp;&nbsp;Predicted class <b>ŷ</b><br><br>
<b>Procedure:</b><br>
&nbsp;&nbsp;&nbsp;<b>For each</b> instance (<b>x</b><sup>(i)</sup>, <b>y</b><sup>(i)</sup>) ∈ <i>A</i><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute the Euclidean distance:
</div>
""", unsafe_allow_html=True)

st.latex(r"d(\mathbf{x}, \mathbf{x}^{(i)}) = \sqrt{\sum_{j=1}^{d} \left(x_j - x_j^{(i)}\right)^2}")

# ✅ CORRIGÉ : color: #1a1a1a ajouté ici aussi
st.markdown("""
<div style="background-color:#fef9e7; border:1px solid #d4ac0d; border-radius:6px; padding:20px; font-family: Arial, sans-serif; margin-top: -10px; color: #1a1a1a;">
&nbsp;&nbsp;&nbsp;Sort the instances in <i>A</i> by increasing distance.<br>
&nbsp;&nbsp;&nbsp;Select the <b>K</b> nearest neighbors.<br>
&nbsp;&nbsp;&nbsp;Perform a majority vote on their classes.<br>
&nbsp;&nbsp;&nbsp;Assign to <b>x</b> the most frequent class among its neighbors.<br><br>
<b>Return</b> the predicted class <b>ŷ</b>
</div>
""", unsafe_allow_html=True)