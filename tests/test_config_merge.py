# Testing merging of configs

cfg_base = \
"""
A:
  B: alpha
  C: beta
  D:
    e: delta
    f: gamma
  G:
    h: epsilon
    i: upsilon
  J:
    K: omicron
    L: 
      M: phi
      N:
        o: psi
        p: yotta
Q:
  R: zeta
  S: tau
  T:
    u: pi
    v: sigma
"""

cfg_extension_a = \
"""
A:
  B: modified_alpha
  C: modified_beta

W:
  X: chi
  Y: eta
  Z:
    a: mu
    b: nu
"""

cfg_extension_b = \
"""
A:
  D:
  G:
    h: modified_epsilon
  J: deleted_key

Q:
  S: modified_tau
  T:
    u: sigma
"""

cfg_base_plus_a = \
"""
A:
  B: modified_alpha
  C: modified_beta
  D:
    e: delta
    f: gamma
  G:
    h: epsilon
    i: upsilon
  J:
    K: omicron
    L: 
      M: phi
      N:
        o: psi
        p: yotta
Q:
  R: zeta
  S: tau
  T:
    u: pi
    v: sigma

W:
  X: chi
  Y: eta
  Z:
    a: mu
    b: nu
"""

cfg_base_plus_b = \
"""
A:
  B: alpha
  C: beta
  D:
  G:
    h: modified_epsilon
  J: deleted_key

Q:
  R: zeta
  S: modified_tau
  T:
    u: sigma
"""


def test_merge_a():
    test_file=  "base_test.yml"
    test_file_a=  "test_a.yml"

    with open(test_file, "w") as wfile:
        wfile.write(cfg_base)

    with open(test_file_a, "w") as wfile:
        wfile.write(cfg_extension_a)
    
    
    import ednaml
    from ednaml.core import EdnaML
    eml = EdnaML()