class CustomInstructTemplate(InstructTemplate):
    template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

    @classmethod
    def format(cls, sample, column_map):
        return cls.template.format(**sample)
