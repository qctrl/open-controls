.. _{{fullname}}:

{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   
   {% block Methods %}
   .. rubric:: Methods
   .. autosummary:: 
      
   {% for item in methods %} 
   {%- if not item.startswith('_') %}
       ~{{ objname }}.{{ item }}
   {%- endif -%}
   {%- endfor -%}
   {% endblock %} 

   {% block Attributes %}
   .. rubric:: Attributes
   .. autosummary::
      
   {% for item in attributes  %}
      ~{{ objname }}.{{ item }}
   {%- endfor -%}
   {% endblock %} 
