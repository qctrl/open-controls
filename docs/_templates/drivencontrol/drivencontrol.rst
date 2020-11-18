.. The custom autosummary implementation for Qctrl.

.. _{{fullname}}:

Driven Control
{{ underline }}

{%
  set module_name =  {"driven_controls": driven_controls,
                       "dynamic_decoupling_sequences": dynamic_decoupling_sequences }
%}

.. autosummary::
   :nosignatures:
   :toctree: .

   {% for item in module_name[objname]   %}
   {%- if not item.startswith('_') %}
       ~{{ module }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% for item in functions %}
   {%- if not item.startswith('_') %}
       ~{{ module.split('.')[0] }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
