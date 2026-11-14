-- Resolve result values from the same snapshot used by every figure renderer.
local file = assert(io.open('generated/report-data.json', 'r'))
local data = quarto.json.decode(file:read('*a'))
file:close()
function Span(el)
  if el.classes:includes('result') then
    local key = el.attributes['key']
    assert(data.values[key], 'Unknown report value: ' .. tostring(key))
    return pandoc.Str(tostring(data.values[key]))
  end
end
