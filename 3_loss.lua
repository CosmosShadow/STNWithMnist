-- loss

criterion = nn.CriterionIntervalWithTerminal(nn.ClassNLLCriterion(), output_terminal)