.. The custom autosummary implementation for Qctrl.

.. _{{fullname}}:

Open Controls
{{ underline }}{{ underline }}

.. autosummary::
   :nosignatures:
   :toctree: {{ module }}

   {% for item in qctrlopencontrols  %}
   {%- if not item.startswith('_') %}
       ~{{ module }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
