.. The custom autosummary implementation for Qctrl.

.. _{{fullname}}:

Q-CTRL Open Controls
{{ underline }}{{ underline }}

.. autosummary::
   :nosignatures:
   :toctree: {{ module }}

   {% for item in qctrlopencontrols  %}
   {%- if not item.startswith('_') %}
       ~{{ module }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% for item in functions %}
   {%- if not item.startswith('_') %}
      ~{{ module }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
